// Copyright 2014 BVLC and contributors.

#include <cstdio>
#include <mpi.h>

#include <boost/lexical_cast.hpp>
#include <protocol/TBinaryProtocol.h>
#include <transport/TSocket.h>
#include <transport/TTransportUtils.h>

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include "Hbase.h"

using std::max;
using std::min;

using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

using namespace apache::hadoop::hbase::thrift;

typedef std::vector<std::string> StrVec;
typedef std::map<std::string,std::string> StrMap;
typedef std::vector<ColumnDescriptor> ColVec;
typedef std::map<std::string,ColumnDescriptor> ColMap;
typedef std::vector<TCell> CellVec;
typedef std::map<std::string,TCell> CellMap;

namespace caffe {

template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param)
    : net_(), test_net_() {
  Init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file)
    : net_(), test_net_() {
  SolverParameter param;
  ReadProtoFromTextFile(param_file, &param);
  Init(param);
}

template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {
  param_ = param;
  if (param_.random_seed() >= 0) {
    Caffe::set_random_seed(param_.random_seed());
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);

  fprintf(stderr, "my rank: %d / %d\n", mpi_rank_, mpi_size_); fflush(stderr);

  NUM_PAR_SRV = 1;
  NUM_DAT_SRV = 1;
  TRAIN_BEGIN = NUM_PAR_SRV;
  TRAIN_END = mpi_size_ - NUM_DAT_SRV;

  // Scaffolding code
  if (mpi_rank_ < NUM_PAR_SRV) {
    LOG(INFO) << "Creating training net.";
    net_.reset(new Net<Dtype>(param_.train_net()));

    if (param_.has_test_net()) {
      LOG(INFO) << "Creating testing net.";
      test_net_.reset(new Net<Dtype>(param_.test_net()));
      CHECK_GT(param_.test_iter(), 0);
      CHECK_GT(param_.test_interval(), 0);
    }
    LOG(INFO) << "** Parameter server initialization done!";
  } else if (mpi_rank_ >= TRAIN_END) {
    LOG(INFO) << "** Data server initialization done!";
  } else {
    LOG(INFO) << "Creating training net.";
    net_.reset(new Net<Dtype>(param_.train_net()));
    LOG(INFO) << "** Trainer " << mpi_rank_ << " initialization done!";
  }

  LOG(INFO) << "Solver scaffolding done.";
}


template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
if (mpi_rank_ < TRAIN_END) {
  Caffe::set_mode(Caffe::Brew(param_.solver_mode()));
  if (param_.solver_mode() == SolverParameter_SolverMode_GPU &&
      param_.has_device_id()) {
    Caffe::SetDevice(param_.device_id());
  }
  Caffe::set_phase(Caffe::TRAIN);
  LOG(INFO) << "Solving " << net_->name();
  PreSolve();

  iter_ = 0;
  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }
}

  if (mpi_rank_ < NUM_PAR_SRV) {
    RunParServer();
  } else if (mpi_rank_ >= TRAIN_END) {
    RunDatServer();
  } else {
    RunTrainer();
  }
}


////////////////////////////////////////////////////////////////
/// Parameter Server ///////////////////////////////////////////
////////////////////////////////////////////////////////////////
template <typename Dtype>
void Solver<Dtype>::RunParServer() {
  // Run a test pass before doing any training to avoid waiting a potentially
  // very long time (param_.test_interval() training iterations) to report that
  // there's not enough memory to run the test net and crash, etc.; and to gauge
  // the effect of the first training iterations.
  if (param_.test_interval()) {
    Test();
  }

  while (iter_++ < param_.max_iter()) {
    for (int i=TRAIN_BEGIN; i<TRAIN_END; i++) {
      // send current parameters to job-i
      net_->SendParams(i);
    }

    for (int i=TRAIN_BEGIN; i<TRAIN_END; i++) {
      // receive update values from job-i
      net_->RecvUpdateValue(i);

      if (param_.display() && iter_ % param_.display() == 0) {
        LOG(INFO) << "Got updates of Iter-" << iter_ << " from node " << i;
      }

      // update the values
      net_->Update();
    }
    if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
      Test();
    }
    // Check if we need to do snapshot
    if (param_.snapshot() && iter_ % param_.snapshot() == 0) {
      Snapshot();
    }
  }
  // After the optimization is done, always do a snapshot.
  iter_--;
  Snapshot();
  LOG(INFO) << "Optimization Done.";
}

////////////////////////////////////////////////////////////////
/// Data Coordinator  //////////////////////////////////////////
////////////////////////////////////////////////////////////////
template <typename Dtype>
void Solver<Dtype>::RunDatServer() {
  std::vector<std::string> keys;
  boost::shared_ptr<TTransport> socket(new TSocket("localhost", 9090));
  boost::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
  boost::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
  shared_ptr<HbaseClient> client(new HbaseClient(protocol));

  try {
    transport->open();
  } catch (const TException &tx) {
    LOG(FATAL) << "ERROR: " << tx.what();
  }

  std::string table = "cifar-train";
  std::string start = "";
  std::vector<std::string> columns;
  std::map<std::string, std::string> attributes;
  columns.push_back("cf:data");

  std::vector<TRowResult> rowResult;

  NetParameter param;
  ReadNetParamsFromTextFileOrDie(param_.train_net(), &param);
  size_t batch_size = -1;
  for (size_t lid=0; lid<param.layers_size(); ++lid) {
    const LayerParameter_LayerType& type = param.layers(lid).type();
    if (type == LayerParameter_LayerType_HBASE_DATA) {
      batch_size = param.layers(lid).data_param().batch_size();
      break;
    }
  }
  if (batch_size < 0) {
    LOG(FATAL) << "ERROR: " << "wrong batch size!";
  }
  size_t num_trainers = TRAIN_END - TRAIN_BEGIN;
  size_t num_fetch = num_trainers * batch_size;
  int scanner = client->scannerOpen(table, start, columns, attributes);
  bool fetch_rows = true;
  int kid = 0;
  while (iter_++ < param_.max_iter()) {
    // fetch fresh data
    if (fetch_rows) {
      client->scannerGetList(rowResult, scanner, batch_size);
      if (rowResult.size() > 0) {
        keys.reserve(keys.size() + rowResult.size());
        for (size_t rid=0; rid<rowResult.size(); ++rid) {
          keys.push_back(rowResult[rid].row);
        }
        DLOG(INFO) << "rows: " << keys.size();
      }
      if (rowResult.size() < batch_size) {
        LOG(INFO) << "Got all row keys, close scanner.";
        fetch_rows = false;
        client->scannerClose(scanner);
      }
    }

    // dispatch data
    for (int i = TRAIN_BEGIN; i < TRAIN_END; ++i) {
      kid %= keys.size();
      int ret = MPI_Send(keys[kid].data(), keys[kid].size(), MPI_CHAR, i, 1, MPI_COMM_WORLD);
      DLOG(INFO) << "Send start key to trainer: " << mpi_rank_ << " -> " << i << "(" << ret << ")";
      DLOG(INFO) << keys[kid].data();
      kid += batch_size;
    }
    for (int i = 0; i < TRAIN_BEGIN; ++i) {
      kid %= keys.size();
      int ret = MPI_Send(keys[kid].data(), keys[kid].size(), MPI_CHAR, i, 1, MPI_COMM_WORLD);
      DLOG(INFO) << "Send start key to trainer: " << mpi_rank_ << " -> " << i << "(" << ret << ")";
      DLOG(INFO) << keys[kid].data();
      kid += batch_size;
    }
  }
}

////////////////////////////////////////////////////////////////
/// Trainers  //////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
template <typename Dtype>
void Solver<Dtype>::RunTrainer() {
  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  vector<Blob<Dtype>*> bottom_vec;

  while (iter_++ < param_.max_iter()) {
    // receive the latest parameters from parameter server
    net_->RecvParams(0);

    Dtype loss = net_->ForwardBackward(bottom_vec);
    ComputeUpdateValue();
    net_->SendUpdateValue(0);

    if (param_.display() && iter_ % param_.display() == 0) {
      LOG(INFO) << "Iteration " << iter_ << ", loss = " << loss
        << " (rank: " << mpi_rank_ << ")";
    }
  }
}

template <typename Dtype>
void Solver<Dtype>::Test() {
  LOG(INFO) << "Iteration " << iter_ << ", Testing net";
  // We need to set phase to test before running.
  Caffe::set_phase(Caffe::TEST);
  CHECK_NOTNULL(test_net_.get())->ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<Blob<Dtype>*> bottom_vec;
  Dtype loss = 0;
  for (int i = 0; i < param_.test_iter(); ++i) {
    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net_->Forward(bottom_vec, &iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter();
    LOG(INFO) << "Test loss: " << loss;
  }
  for (int i = 0; i < test_score.size(); ++i) {
    LOG(INFO) << "Test score #" << i << ": "
        << test_score[i] / param_.test_iter();
  }
  Caffe::set_phase(Caffe::TRAIN);
}


template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  NetParameter net_param;
  // For intermediate results, we will also dump the gradient values.
  net_->ToProto(&net_param, param_.snapshot_diff());
  string filename(param_.snapshot_prefix());
  const int kBufferSize = 20;
  char iter_str_buffer[kBufferSize];
  snprintf(iter_str_buffer, kBufferSize, "_iter_%d", iter_);
  filename += iter_str_buffer;
  LOG(INFO) << "Snapshotting to " << filename;
  WriteProtoToBinaryFile(net_param, filename.c_str());
  SolverState state;
  SnapshotSolverState(&state);
  state.set_iter(iter_);
  state.set_learned_net(filename);
  filename += ".solverstate";
  LOG(INFO) << "Snapshotting solver state to " << filename;
  WriteProtoToBinaryFile(state, filename.c_str());
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  SolverState state;
  NetParameter net_param;
  ReadProtoFromBinaryFile(state_file, &state);
  if (state.has_learned_net()) {
    ReadProtoFromBinaryFile(state.learned_net().c_str(), &net_param);
    net_->CopyTrainedLayersFrom(net_param);
  }
  iter_ = state.iter();
  RestoreSolverState(state);
}


// Return the current learning rate. The currently implemented learning rate
// policies are as follows:
//    - fixed: always return base_lr.
//    - step: return base_lr * gamma ^ (floor(iter / step))
//    - exp: return base_lr * gamma ^ iter
//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
// where base_lr, gamma, step and power are defined in the solver parameter
// protocol buffer, and iter is the current iteration.
template <typename Dtype>
Dtype SGDSolver<Dtype>::GetLearningRate() {
  Dtype rate;
  const string& lr_policy = this->param_.lr_policy();
  if (lr_policy == "fixed") {
    rate = this->param_.base_lr();
  } else if (lr_policy == "step") {
    int current_step = this->iter_ / this->param_.stepsize();
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), current_step);
  } else if (lr_policy == "exp") {
    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
  } else if (lr_policy == "inv") {
    rate = this->param_.base_lr() *
        pow(Dtype(1) + this->param_.gamma() * this->iter_,
            - this->param_.power());
  } else {
    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }
  return rate;
}


template <typename Dtype>
void SGDSolver<Dtype>::PreSolve() {
  // Initialize the history
  vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  history_.clear();
  for (int i = 0; i < net_params.size(); ++i) {
    const Blob<Dtype>* net_param = net_params[i].get();
    history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(
        net_param->num(), net_param->channels(), net_param->height(),
        net_param->width())));
  }
}


template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue() {
  vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  vector<float>& net_params_lr = this->net_->params_lr();
  vector<float>& net_params_weight_decay = this->net_->params_weight_decay();
  // get the learning rate
  Dtype rate = GetLearningRate();
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
  }
  Dtype momentum = this->param_.momentum();
  Dtype weight_decay = this->param_.weight_decay();
  switch (Caffe::mode()) {
  case Caffe::CPU:
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // Compute the value to history, and then copy them to the blob's diff.
      Dtype local_rate = rate * net_params_lr[param_id];
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
      caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
          net_params[param_id]->cpu_diff(), momentum,
          history_[param_id]->mutable_cpu_data());
      if (local_decay) {
        // add weight decay
        caffe_axpy(net_params[param_id]->count(),
            local_decay * local_rate,
            net_params[param_id]->cpu_data(),
            history_[param_id]->mutable_cpu_data());
      }
      // copy
      caffe_copy(net_params[param_id]->count(),
          history_[param_id]->cpu_data(),
          net_params[param_id]->mutable_cpu_diff());
    }
    break;
  case Caffe::GPU:
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // Compute the value to history, and then copy them to the blob's diff.
      Dtype local_rate = rate * net_params_lr[param_id];
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
      caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
          net_params[param_id]->gpu_diff(), momentum,
          history_[param_id]->mutable_gpu_data());
      if (local_decay) {
        // add weight decay
        caffe_gpu_axpy(net_params[param_id]->count(),
            local_decay * local_rate,
            net_params[param_id]->gpu_data(),
            history_[param_id]->mutable_gpu_data());
      }
      // copy
      caffe_gpu_copy(net_params[param_id]->count(),
          history_[param_id]->gpu_data(),
          net_params[param_id]->mutable_gpu_diff());
    }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverState(SolverState* state) {
  state->clear_history();
  for (int i = 0; i < history_.size(); ++i) {
    // Add history
    BlobProto* history_blob = state->add_history();
    history_[i]->ToProto(history_blob);
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverState(const SolverState& state) {
  CHECK_EQ(state.history_size(), history_.size())
      << "Incorrect length of history blobs.";
  LOG(INFO) << "SGDSolver: restoring history";
  for (int i = 0; i < history_.size(); ++i) {
    history_[i]->FromProto(state.history(i));
  }
}

INSTANTIATE_CLASS(Solver);
INSTANTIATE_CLASS(SGDSolver);

}  // namespace caffe
