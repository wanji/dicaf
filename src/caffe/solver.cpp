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
    : net_() {
  Init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file)
    : net_() {
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
  if (param_.has_host() && param_.has_port()) {
    Caffe::set_hbase(param_.host(), param_.port());
  }

  device_count_ = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);

  LOG(INFO) << "My Rank: " << mpi_rank_ << " / " << mpi_size_;

  train_begin_ = NUM_PAR_SRV;
  train_end_ = mpi_size_ - NUM_DAT_SRV;

  // Scaffolding code
  if (mpi_rank_ < NUM_PAR_SRV) {
    Caffe::set_phase(Caffe::TEST);

    LOG(INFO) << "Creating testing net ...";
    if (param_.has_test_net()) {
      net_.reset(new Net<Dtype>(param_.test_net()));
      CHECK_GT(param_.test_iter(), 0);
      CHECK_GT(param_.test_interval(), 0);
    } else {
      net_.reset(new Net<Dtype>(param_.train_net()));
      LOG(ERROR) << "** Cannot find test net, using training net to host the parameters";
    }
    LOG(INFO) << "** Parameter server initialization done!";

  } else if (mpi_rank_ >= train_end_) {

    LOG(INFO) << "** Data server initialization done!";

  } else {
    Caffe::set_phase(Caffe::TRAIN);

    LOG(INFO) << "Creating training net ...";
    net_.reset(new Net<Dtype>(param_.train_net()));
    LOG(INFO) << "** Trainer " << mpi_rank_ << " initialization done!";

  }

  LOG(INFO) << "Solver scaffolding done.";
}


template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
  if (mpi_rank_ < train_end_) {
    Caffe::set_mode(Caffe::Brew(param_.solver_mode()));
    // if (param_.solver_mode() == SolverParameter_SolverMode_GPU &&
    //     param_.has_device_id()) {
    //   Caffe::SetDevice(param_.device_id());
    // }
    if (param_.solver_mode() == SolverParameter_SolverMode_GPU) {
      cudaError_t error_id = cudaGetDeviceCount(&device_count_);

      if (error_id != cudaSuccess) {
          LOG(FATAL) << "cudaGetDeviceCount returned " << error_id << ":"
            << cudaGetErrorString(error_id);
      }
      
      if (device_count_ == 0) {
          LOG(FATAL) << "There are no available device(s) that support CUDA";
      }


      if (mpi_rank_ < train_begin_) {
        LOG(INFO) << "rank " << mpi_rank_ << " adopt device " << 0;
        Caffe::SetDevice(0);
      } else {
        int device_id = (mpi_rank_ - train_begin_) % device_count_;
        LOG(INFO) << "rank " << mpi_rank_ << " adopt device " << device_id;
        Caffe::SetDevice(device_id);
      }
    }
  }

  if (mpi_rank_ < NUM_PAR_SRV) {
    LOG(INFO) << "Solving " << net_->name();

    iter_ = 0;
    if (resume_file) {
      LOG(INFO) << "Restoring previous solver status from " << resume_file;
      Restore(resume_file);
    }

    if (0 == mpi_rank_) {
      for (int i=train_begin_; i<mpi_size_; i++) {
        int ret = MPI_Send(&iter_, 1, MPI_INT, i, MPI_TAG_INIT_ITER, MPI_COMM_WORLD);
        if (ret != MPI_SUCCESS) {
          LOG(FATAL) << "Sending iter_ to trainers " << i << " Failed!";
        }
      }
    }

    RunParServer();
  } else if (mpi_rank_ < train_end_) {
    MPI_Status stat;
    int ret = MPI_Recv(&iter_, 1, MPI_INT, 0, MPI_TAG_INIT_ITER, MPI_COMM_WORLD, &stat);
    if (ret != MPI_SUCCESS) {
      LOG(FATAL) << "Receiving iter_ from header Failed! rank=" << mpi_rank_;
    }

    LOG(INFO) << "RunTrainer ...";
    PreSolve();
    RunTrainer();
  } else {
    MPI_Status stat;
    int ret = MPI_Recv(&iter_, 1, MPI_INT, 0, MPI_TAG_INIT_ITER, MPI_COMM_WORLD, &stat);
    if (ret != MPI_SUCCESS) {
      LOG(FATAL) << "Receiving iter_ from header Failed! rank=" << mpi_rank_;
    }

    LOG(INFO) << "RunDatServer ...";
    RunDatServer();
  }
  LOG(INFO) << "Optimization Done. (rank: " << mpi_rank_ << ")";
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
    for (int i=train_begin_; i<train_end_; i++) {
      DLOG(INFO) << "Parameters Sending: [" << mpi_rank_ << "] -> [" << i << "] " 
        << "(iter: " << iter_ << "/" << param_.max_iter() << ")";
      // send current parameters to job-i
      net_->SendParams(i);
      DLOG(INFO) << "Parameters Sent:    [" << mpi_rank_ << "] -> [" << i << "] " 
        << "(iter: " << iter_ << "/" << param_.max_iter() << ")";
    }
    //MPI_Barrier(MPI_COMM_WORLD);

    for (int i=train_begin_; i<train_end_; i++) {
      DLOG(INFO) << "Updates Receiving: [" << mpi_rank_ << "] <- [" << i << "] " 
        << "(iter: " << iter_ << "/" << param_.max_iter() << ")";
      // receive update values from job-i
      net_->RecvUpdateValue(i);
      DLOG(INFO) << "Updates Received:  [" << mpi_rank_ << "] <- [" << i << "] " 
        << "(iter: " << iter_ << "/" << param_.max_iter() << ")";

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
    DLOG(INFO) << "Parameters Receiving: [" << mpi_rank_ << "] <- [0] " 
      << "(iter: " << iter_ << "/" << param_.max_iter() << ")";
    // receive the latest parameters from parameter server
    net_->RecvParams(0);
    DLOG(INFO) << "Parameters Received:  [" << mpi_rank_ << "] <- [0] " 
      << "(iter: " << iter_ << "/" << param_.max_iter() << ")";

    Dtype loss = net_->ForwardBackward(bottom_vec);
    ComputeUpdateValue();

    DLOG(INFO) << "Updates Sending: [" << mpi_rank_ << "] -> [0] " 
      << "(iter: " << iter_ << "/" << param_.max_iter() << ")";
    net_->SendUpdateValue(0);
    DLOG(INFO) << "Updates Sent:    [" << mpi_rank_ << "] -> [0] " 
      << "(iter: " << iter_ << "/" << param_.max_iter() << ")";

    if (param_.display() && iter_ % param_.display() == 0) {
      LOG(INFO) << "Iteration " << iter_ << ", loss = " << loss
        << " (rank: " << mpi_rank_ << ")";
    }
  }
}

////////////////////////////////////////////////////////////////
/// Data Coordinator  //////////////////////////////////////////
////////////////////////////////////////////////////////////////
template <typename Dtype>
void Solver<Dtype>::RunDatServer() {

  // parse network parameters
  NetParameter train_param;
  ReadNetParamsFromTextFileOrDie(param_.train_net(), &train_param);

  // get training batch size
  size_t train_batch_size = -1;
  size_t train_fetch = -1;
  std::string table;
  for (size_t lid=0; lid<train_param.layers_size(); ++lid) {
    const LayerParameter_LayerType& type = train_param.layers(lid).type();
    if (type == LayerParameter_LayerType_HBASE_DATA) {
      train_batch_size = train_param.layers(lid).data_param().batch_size();
      table = train_param.layers(lid).data_param().source();
      break;
    }
  }
  if (train_batch_size < 0) {
    LOG(FATAL) << "ERROR: " << "wrong train batch size!";
  }
  train_fetch = (train_end_ - train_begin_) * train_batch_size;

  LOG(INFO) << "- Training batch size: " << train_batch_size;
  LOG(INFO) << "- Fetch " << train_fetch << " row keys for training per iter.";

  LOG(INFO) << "Connecting to HBase ...";
  boost::shared_ptr<TTransport> socket(new TSocket(Caffe::host(), Caffe::port()));
  boost::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
  boost::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
  shared_ptr<HbaseClient> client(new HbaseClient(protocol));
  try {
    transport->open();
  } catch (const TException &tx) {
    LOG(FATAL) << "ERROR: " << tx.what();
  }
  LOG(INFO) << "HBase connection established!";

  // table information
  std::vector<std::string> columns;
  std::map<std::string, std::string> attributes;
  columns.push_back("cf:data");

  // store the scanned results
  std::vector<TRowResult> rowResult;

  // fetch row keys from HBase
  bool train_fetch_rows = true;
  int train_kid = 0;
  std::vector<std::string> train_keys;

  // initialize the scanners
  int train_scanner = client->scannerOpen(table, "", columns, attributes);

#if PREFETCH_ROW_KEYS
  size_t fetch_num = 1024;

  do {
      client->scannerGetList(rowResult, train_scanner, fetch_num);
      if (rowResult.size() > 0) {
        train_keys.reserve(train_keys.size() + rowResult.size());
        for (size_t rid=0; rid<rowResult.size(); ++rid) {
          train_keys.push_back(rowResult[rid].row);
        }
      }
      LOG(INFO) << rowResult.size() << " training row keys fetched!";
  } while (rowResult.size() == fetch_num);
  LOG(INFO) << "Got all training row keys (" << train_keys.size()
    << "), close scanner.";
  client->scannerClose(train_scanner);
#endif

  DLOG(INFO) << "iter/max_iter: " << iter_ << "/" << param_.max_iter();

  bool first_iter = true;
  while (iter_++ < param_.max_iter()) {
DLOG(INFO) << "RunDatServer_iter: " << iter_ << "/" << param_.max_iter() << " (rank: " << mpi_rank_ << ")";
#if !PREFETCH_ROW_KEYS
    // fetch fresh training data
    if (train_fetch_rows) {
      client->scannerGetList(rowResult, train_scanner, train_fetch);
      if (rowResult.size() > 0) {
        train_keys.reserve(train_keys.size() + rowResult.size());
        for (size_t rid=0; rid<rowResult.size(); ++rid) {
          train_keys.push_back(rowResult[rid].row);
        }
      }
      if (rowResult.size() < train_fetch) {
        LOG(INFO) << "Got all training row keys (" << train_keys.size()
          << "), close scanner.";
        train_fetch_rows = false;
        client->scannerClose(train_scanner);
      }
    } // end if
DLOG(INFO) << "RunDatServer_iter-2: " << iter_ << "/" << param_.max_iter() << " (rank: " << mpi_rank_ << ")";
#endif

    // dispatch training data
    for (int i = train_begin_; i < train_end_; ++i) {
      train_kid %= train_keys.size();
      DLOG(INFO) << "Sending start key to trainer: "
        << mpi_rank_ << " -> " << i
        << " " << train_kid << ": " << train_keys.size();

      int ret = MPI_Send(train_keys[train_kid].data(),
          train_keys[train_kid].size(), MPI_CHAR, i, MPI_TAG_DATA_TRAIN, MPI_COMM_WORLD);

      if (MPI_SUCCESS != ret) {
        LOG(FATAL) << "MPI_Recv failed";
      }

      DLOG(INFO) << "Sent: " << ret << "(" << mpi_rank_ << " -> " << i << ") "
        << train_keys[train_kid];
      train_kid += train_batch_size;
    } // end for

DLOG(INFO) << "RunDatServer_iter-1: " << iter_ << "/" << param_.max_iter() << " (rank: " << mpi_rank_ << ")";
  } // end while

  DLOG(INFO) << "Post processing";

  for (int i = train_begin_; i < train_end_; ++i) {
    int ret = MPI_Send(MPI_MSG_END_DATA_PREFETCH,
        sizeof(MPI_MSG_END_DATA_PREFETCH), MPI_CHAR, i, MPI_TAG_DATA_TRAIN, MPI_COMM_WORLD);

    if (MPI_SUCCESS != ret) {
      LOG(FATAL) << "MPI_Recv failed";
    }
  } // end for

  DLOG(INFO) << "Post processing done!";

  try {
    transport->close();
  } catch (const TException &tx) {
    LOG(FATAL) << "ERROR: " << tx.what();
  }
  LOG(INFO) << "HBase connection closed!";
}

template <typename Dtype>
void Solver<Dtype>::Test() {
  LOG(INFO) << "Iteration " << iter_ << ", Testing net";
  // We need to set phase to test before running.
  // no need any more - wj (2014-06-20)
  // Caffe::set_phase(Caffe::TEST);
  // CHECK_NOTNULL(test_net_.get())->ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<Blob<Dtype>*> bottom_vec;
  Dtype loss = 0;
  for (int i = 0; i < param_.test_iter(); ++i) {
    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        net_->Forward(bottom_vec, &iter_loss);
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
  // Caffe::set_phase(Caffe::TRAIN);
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
