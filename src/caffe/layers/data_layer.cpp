// Copyright 2014 BVLC and contributors.

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>
#include <mpi.h>

#include <boost/lexical_cast.hpp>
#include <protocol/TBinaryProtocol.h>
#include <transport/TSocket.h>
#include <transport/TTransportUtils.h>

#include <string>
#include <vector>
#include <sstream>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

#include "Hbase.h"

using std::string;

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
void* DataLayerPrefetch(void* layer_pointer) {
  CHECK(layer_pointer);
  DataLayer<Dtype>* layer = static_cast<DataLayer<Dtype>*>(layer_pointer);
  CHECK(layer);
  Datum datum;
  CHECK(layer->prefetch_data_);
  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  Dtype* top_label;
  if (layer->output_labels_) {
    top_label = layer->prefetch_label_->mutable_cpu_data();
  }
  const Dtype scale = layer->layer_param_.data_param().scale();
  const int batch_size = layer->layer_param_.data_param().batch_size();
  const int crop_size = layer->layer_param_.data_param().crop_size();
  const bool mirror = layer->layer_param_.data_param().mirror();

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
        << "set at the same time.";
  }
  // datum scales
  const int channels = layer->datum_channels_;
  const int height = layer->datum_height_;
  const int width = layer->datum_width_;
  const int size = layer->datum_size_;
  const Dtype* mean = layer->data_mean_.cpu_data();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    CHECK(layer->iter_);
    CHECK(layer->iter_->Valid());
    datum.ParseFromString(layer->iter_->value().ToString());
    const string& data = datum.data();
    if (crop_size) {
      CHECK(data.size()) << "Image cropping only support uint8 data";
      int h_off, w_off;
      // We only do random crop when we do training.
      if (layer->phase_ == Caffe::TRAIN) {
        h_off = layer->PrefetchRand() % (height - crop_size);
        w_off = layer->PrefetchRand() % (width - crop_size);
      } else {
        h_off = (height - crop_size) / 2;
        w_off = (width - crop_size) / 2;
      }
      if (mirror && layer->PrefetchRand() % 2) {
        // Copy mirrored version
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < crop_size; ++h) {
            for (int w = 0; w < crop_size; ++w) {
              int top_index = ((item_id * channels + c) * crop_size + h)
                              * crop_size + (crop_size - 1 - w);
              int data_index = (c * height + h + h_off) * width + w + w_off;
              Dtype datum_element =
                  static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
              top_data[top_index] = (datum_element - mean[data_index]) * scale;
            }
          }
        }
      } else {
        // Normal copy
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < crop_size; ++h) {
            for (int w = 0; w < crop_size; ++w) {
              int top_index = ((item_id * channels + c) * crop_size + h)
                              * crop_size + w;
              int data_index = (c * height + h + h_off) * width + w + w_off;
              Dtype datum_element =
                  static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
              top_data[top_index] = (datum_element - mean[data_index]) * scale;
            }
          }
        }
      }
    } else {
      // we will prefer to use data() first, and then try float_data()
      if (data.size()) {
        for (int j = 0; j < size; ++j) {
          Dtype datum_element =
              static_cast<Dtype>(static_cast<uint8_t>(data[j]));
          top_data[item_id * size + j] = (datum_element - mean[j]) * scale;
        }
      } else {
        for (int j = 0; j < size; ++j) {
          top_data[item_id * size + j] =
              (datum.float_data(j) - mean[j]) * scale;
        }
      }
    }

    if (layer->output_labels_) {
      top_label[item_id] = datum.label();
    }
    // go to the next iter
    layer->iter_->Next();
    if (!layer->iter_->Valid()) {
      // We have reached the end. Restart from the first.
      int rank = -1;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      DLOG(INFO) << "Restarting data prefetching from start. (rank " << rank << ")";
      layer->iter_->SeekToFirst();
    }
  }

  return static_cast<void*>(NULL);
}

template <typename Dtype>
void* HBaseDataLayerPrefetch(void* layer_pointer) {
  CHECK(layer_pointer);
  HBaseDataLayer<Dtype>* layer = static_cast<HBaseDataLayer<Dtype>*>(layer_pointer);
  CHECK(layer);
  Datum datum;
  CHECK(layer->prefetch_data_);
  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  Dtype* top_label;
  if (layer->output_labels_) {
    top_label = layer->prefetch_label_->mutable_cpu_data();
  }
  const Dtype scale = layer->layer_param_.data_param().scale();
  const int batch_size = layer->layer_param_.data_param().batch_size();
  const int crop_size = layer->layer_param_.data_param().crop_size();
  const bool mirror = layer->layer_param_.data_param().mirror();

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
        << "set at the same time.";
  }
  // datum scales
  const int channels = layer->datum_channels_;
  const int height = layer->datum_height_;
  const int width = layer->datum_width_;
  const int size = layer->datum_size_;
  const Dtype* mean = layer->data_mean_.cpu_data();

  int mpi_rank;
  int mpi_size;
  char start_buf[256];
  MPI_Status stat;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  DLOG(INFO) << "Waiting for start key from data coordinator: "
    << mpi_rank << " <- " << mpi_size - 1;
  MPI_Recv(start_buf, sizeof(start_buf) / sizeof(start_buf[0]), MPI_CHAR, mpi_size - 1, 1, MPI_COMM_WORLD, &stat);
  DLOG(INFO) << "Recveive start key from data coordinator: "
    << mpi_rank << " <- " << mpi_size - 1 << " (" << start_buf << ")";

  // fetch data from HBase
//int scanner = layer->client_->scannerOpen(layer->table_, layer->start_,
  int scanner = layer->client_->scannerOpen(layer->table_, start_buf,
      layer->columns_, layer->attributes_);
  unsigned int num_get = batch_size + 1;
  std::vector<TRowResult> data_batch;
  data_batch.reserve(num_get);
  std::vector<TRowResult> rowResult;
// DLOG(INFO) << "out : " << data_batch.size() << "/" << num_get << ", rank: " << rank;
  while (data_batch.size() < num_get) {
// DLOG(INFO) << data_batch.size() << "/" << num_get << ", rank: " << rank;
    size_t rest = num_get - data_batch.size();
    layer->client_->scannerGetList(rowResult, scanner, rest);
    if (rowResult.size() < rest) {
      // We have reached the end. Restart from the first.
// DLOG(INFO) << rowResult.size() << "/" << rest << ", rank: " << rank;
      layer->ResetScanner();
      scanner = layer->client_->scannerOpen(layer->table_, layer->start_,
      layer->columns_, layer->attributes_);
    }
    data_batch.insert(data_batch.end(), rowResult.begin(), rowResult.end());
  }
// DLOG(INFO) << "out" << ", rank: " << rank;
  layer->start_ = rowResult[rowResult.size() - 1].row;
  layer->client_->scannerClose(scanner);

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    datum.ParseFromString(data_batch[item_id].columns.begin()->second.value);
    const string& data = datum.data();
    if (crop_size) {
      CHECK(data.size()) << "Image cropping only support uint8 data";
      int h_off, w_off;
      // We only do random crop when we do training.
      if (layer->phase_ == Caffe::TRAIN) {
        h_off = layer->PrefetchRand() % (height - crop_size);
        w_off = layer->PrefetchRand() % (width - crop_size);
      } else {
        h_off = (height - crop_size) / 2;
        w_off = (width - crop_size) / 2;
      }
      if (mirror && layer->PrefetchRand() % 2) {
        // Copy mirrored version
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < crop_size; ++h) {
            for (int w = 0; w < crop_size; ++w) {
              int top_index = ((item_id * channels + c) * crop_size + h)
                              * crop_size + (crop_size - 1 - w);
              int data_index = (c * height + h + h_off) * width + w + w_off;
              Dtype datum_element =
                  static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
              top_data[top_index] = (datum_element - mean[data_index]) * scale;
            }
          }
        }
      } else {
        // Normal copy
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < crop_size; ++h) {
            for (int w = 0; w < crop_size; ++w) {
              int top_index = ((item_id * channels + c) * crop_size + h)
                              * crop_size + w;
              int data_index = (c * height + h + h_off) * width + w + w_off;
              Dtype datum_element =
                  static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
              top_data[top_index] = (datum_element - mean[data_index]) * scale;
            }
          }
        }
      }
    } else {
      // we will prefer to use data() first, and then try float_data()
      if (data.size()) {
        for (int j = 0; j < size; ++j) {
          Dtype datum_element =
              static_cast<Dtype>(static_cast<uint8_t>(data[j]));
          top_data[item_id * size + j] = (datum_element - mean[j]) * scale;
        }
      } else {
        for (int j = 0; j < size; ++j) {
          top_data[item_id * size + j] =
              (datum.float_data(j) - mean[j]) * scale;
        }
      }
    }

    if (layer->output_labels_) {
      top_label[item_id] = datum.label();
    }
  }

  return static_cast<void*>(NULL);
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer<Dtype>() {
  JoinPrefetchThread();
}

template <typename Dtype>
void DataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 0) << "Data Layer takes no input blobs.";
  CHECK_GE(top->size(), 1) << "Data Layer takes at least one blob as output.";
  CHECK_LE(top->size(), 2) << "Data Layer takes at most two blobs as output.";
  if (top->size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }

  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(this->SetUpDB());

  // image
  int crop_size = this->layer_param_.data_param().crop_size();
  if (crop_size > 0) {
    (*top)[0]->Reshape(this->layer_param_.data_param().batch_size(),
                       datum.channels(), crop_size, crop_size);
    prefetch_data_.reset(new Blob<Dtype>(
        this->layer_param_.data_param().batch_size(), datum.channels(),
        crop_size, crop_size));
  } else {
    (*top)[0]->Reshape(
        this->layer_param_.data_param().batch_size(), datum.channels(),
        datum.height(), datum.width());
    prefetch_data_.reset(new Blob<Dtype>(
        this->layer_param_.data_param().batch_size(), datum.channels(),
        datum.height(), datum.width()));
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  if (output_labels_) {
    (*top)[1]->Reshape(this->layer_param_.data_param().batch_size(), 1, 1, 1);
    prefetch_label_.reset(
        new Blob<Dtype>(this->layer_param_.data_param().batch_size(), 1, 1, 1));
  }
  // datum size
  datum_channels_ = datum.channels();
  datum_height_ = datum.height();
  datum_width_ = datum.width();
  datum_size_ = datum.channels() * datum.height() * datum.width();
  CHECK_GT(datum_height_, crop_size);
  CHECK_GT(datum_width_, crop_size);
  // check if we want to have mean
  if (this->layer_param_.data_param().has_mean_file()) {
    const string& mean_file = this->layer_param_.data_param().mean_file();
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.channels(), datum_channels_);
    CHECK_EQ(data_mean_.height(), datum_height_);
    CHECK_EQ(data_mean_.width(), datum_width_);
  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, datum_channels_, datum_height_, datum_width_);
  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_->mutable_cpu_data();
  if (output_labels_) {
    prefetch_label_->mutable_cpu_data();
  }
  data_mean_.cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
std::string DataLayer<Dtype>::SetUpDB() {
  // Initialize the leveldb
  leveldb::DB* db_temp;
  leveldb::Options options;
  options.create_if_missing = false;
  options.max_open_files = 100;
  std::ostringstream data_source;
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  data_source << this->layer_param_.data_param().source() << "-" << rank;
  LOG(INFO) << "Opening leveldb " << data_source.str();
  leveldb::Status status = leveldb::DB::Open(
      options, data_source.str(), &db_temp);
  CHECK(status.ok()) << "Failed to open leveldb "
      << data_source.str() << std::endl
      << status.ToString();
  db_.reset(db_temp);
  iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
  iter_->SeekToFirst();
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      iter_->Next();
      if (!iter_->Valid()) {
        iter_->SeekToFirst();
      }
    }
  }
  return iter_->value().ToString();
}

template <typename Dtype>
void DataLayer<Dtype>::CreatePrefetchThread() {
  phase_ = Caffe::phase();
  const bool prefetch_needs_rand = (phase_ == Caffe::TRAIN) &&
      (this->layer_param_.data_param().mirror() ||
       this->layer_param_.data_param().crop_size());
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }
  // Create the thread.
  CHECK(!pthread_create(&thread_, NULL, DataLayerPrefetch<Dtype>,
        static_cast<void*>(this))) << "Pthread execution failed.";
}

template <typename Dtype>
void DataLayer<Dtype>::JoinPrefetchThread() {
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
}

template <typename Dtype>
unsigned int DataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
Dtype DataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_->count(), prefetch_data_->cpu_data(),
             (*top)[0]->mutable_cpu_data());
  if (output_labels_) {
    caffe_copy(prefetch_label_->count(), prefetch_label_->cpu_data(),
               (*top)[1]->mutable_cpu_data());
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
  return Dtype(0.);
}

INSTANTIATE_CLASS(DataLayer);

template <typename Dtype>
std::string HBaseDataLayer<Dtype>::SetUpDB() {
	// Initialize the HBase client
	boost::shared_ptr<TTransport> socket(new TSocket(
        this->layer_param_.data_param().host(),
        this->layer_param_.data_param().port()));
	boost::shared_ptr<TTransport> transport(new TBufferedTransport(socket));
	boost::shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));
	this->client_.reset(new HbaseClient(protocol));

  this->table_ = this->layer_param_.data_param().source();
  this->start_ = "";
  this->columns_.push_back("cf:data");

  std::vector<TRowResult> rowResult;

	try {
		transport->open();

		// fetch the first row
    LOG(INFO) << "** hbase info (host):  " << this->layer_param_.data_param().host()
      << ":" << this->layer_param_.data_param().port()
		  << "/" << this->table_;

    ResetScanner();
    this->client_->getRow(rowResult, this->table_, this->start_, this->attributes_);
    if (rowResult.size() < 1) {
      LOG(FATAL) << "Empty database!";
    } else if (rowResult.size() > 1) {
      LOG(FATAL) << "Unknown database error!";
    }
	} catch (const TException &tx) {
		LOG(FATAL) << "ERROR: " << tx.what();
  }

	// Check if we would need to randomly skip a few data points
	if (this->layer_param_.data_param().rand_skip()) {
	  unsigned int skip = caffe_rng_rand() %
	                      this->layer_param_.data_param().rand_skip();
	  LOG(INFO) << "Skipping first " << skip << " data points.";
		int scanner = this->client_->scannerOpen(this->table_, this->start_,
        this->columns_, this->attributes_);
    // fetch skip+1 rows, and use the key of the `skip+1`-th row as new start_
    unsigned int num_get = skip + 1;
    while (num_get > 0) {
      this->client_->scannerGetList(rowResult, scanner, num_get);
      if (rowResult.size() < num_get) {
        ResetScanner();
        scanner = this->client_->scannerOpen(this->table_, this->start_,
            this->columns_, this->attributes_);
      }
      num_get -= rowResult.size();
    }
    this->start_ = rowResult[rowResult.size() - 1].row;
    this->client_->scannerClose(scanner);
	}
  return rowResult[0].columns.begin()->second.value;
}

template <typename Dtype>
void HBaseDataLayer<Dtype>::ResetScanner() {
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  DLOG(INFO) << "Restarting data prefetching from start. (rank " << rank << ")";
  this->start_ = "";
}


template <typename Dtype>
void HBaseDataLayer<Dtype>::CreatePrefetchThread() {
  this->phase_ = Caffe::phase();
  const bool prefetch_needs_rand = (this->phase_ == Caffe::TRAIN) &&
      (this->layer_param_.data_param().mirror() ||
       this->layer_param_.data_param().crop_size());
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    this->prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    this->prefetch_rng_.reset();
  }
  // Create the thread.
  CHECK(!pthread_create(&this->thread_, NULL, HBaseDataLayerPrefetch<Dtype>,
        static_cast<void*>(this))) << "Pthread execution failed.";
}

INSTANTIATE_CLASS(HBaseDataLayer);

}  // namespace caffe
