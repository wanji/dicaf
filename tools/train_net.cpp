// Copyright 2014 BVLC and contributors.
//
// This is a simple script that allows one to quickly train a network whose
// parameters are specified by text format protocol buffers.
// Usage:
//    train_net net_proto_file solver_proto_file [resume_point_file]

#include <cuda_runtime.h>
#include <mpi.h>

#include <cstring>

#include "caffe/caffe.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 2 || argc > 3) {
    LOG(ERROR) << "Usage: train_net solver_proto_file [resume_point_file]";
    return 1;
  }

  int provided;
  int ret = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (ret != MPI_SUCCESS) {
    LOG(FATAL) << "** MPI_Init Failed: " << mpi_get_err_str(ret);
  }
  if (provided != MPI_THREAD_MULTIPLE) {
    LOG(FATAL) << "** MPI_THREAD_MULTIPLE not meet: " << provided;
  }

  SolverParameter solver_param;
  ReadProtoFromTextFileOrDie(argv[1], &solver_param);

  SGDSolver<float> solver(solver_param);
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (argc == 3) {
    LOG(INFO) << "Resuming from " << argv[2] << " (rank: " << mpi_rank << ")";
    solver.Solve(argv[2]);
  } else {
    LOG(INFO) << "New optimization task" << " (rank: " << mpi_rank << ")";
    solver.Solve();
  }
  DLOG(INFO) << "Optimization (DBG).";

  MPI_Finalize();
  return 0;
}
