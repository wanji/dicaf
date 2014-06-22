#!/usr/bin/env sh

TOOLS=`pwd`/../../build/tools

MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 GLOG_logtostderr=1 \
mpirun -hostfile hosts -n 6 -wdir `pwd` \
$TOOLS/train_net.bin cifar10_quick_solver.prototxt

#reduce learning rate by fctor of 10 after 8 epochs
MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 GLOG_logtostderr=1 \
mpirun -hostfile hosts -n 6 -wdir `pwd` \
$TOOLS/train_net.bin cifar10_quick_solver_lr1.prototxt cifar10_quick_iter_1000.solverstate
