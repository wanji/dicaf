#!/usr/bin/env sh

TOOLS=../../build/tools

# MV4_DEBUG_CORESIZE=unlimited MV2_DEBUG_SHOW_BACKTRACE=1
MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 GLOG_logtostderr=1 mpirun -n 6 $TOOLS/train_net.bin cifar10_quick_solver.prototxt

#reduce learning rate by fctor of 10 after 8 epochs
MV2_USE_CUDA=1 GLOG_logtostderr=1 mpirun -n 6 $TOOLS/train_net.bin cifar10_quick_solver_lr1.prototxt cifar10_quick_iter_4000.solverstate
