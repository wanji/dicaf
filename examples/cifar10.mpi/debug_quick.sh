#!/usr/bin/env sh

TOOLS=`pwd`/../../build/tools

MV2_DEBUG_CORESIZE=unlimited MV2_DEBUG_SHOW_BACKTRACE=1 \
MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 GLOG_logtostderr=1 \
mpirun -hostfile hosts -n 6 -wdir `pwd` \
xterm -e gdb --args \
$TOOLS/train_net.bin cifar10_quick_solver.prototxt

