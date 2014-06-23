#!/usr/bin/env sh

TOOLS=`pwd`/../../build/tools

NJOBS=6
N_TRAIN_JOBS=`expr $NJOBS - 2`
N_FIRST_ITER=`expr 4000 / $N_TRAIN_JOBS`
N_SECOND_ITER=`expr 5000 / $N_TRAIN_JOBS`

cat cifar10_quick_solver.prototxt.template | sed "s/FIRST_ITER/$N_FIRST_ITER/g" > cifar10_quick_solver.prototxt
cat cifar10_quick_solver_lr1.prototxt.template | sed "s/SECOND_ITER/$N_SECOND_ITER/g" > cifar10_quick_solver_lr1.prototxt

MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 GLOG_logtostderr=1 \
srun -n $NJOBS -p gpu --gres=gpu:3 --ntasks-per-node=3 \
$TOOLS/train_net.bin cifar10_quick_solver.prototxt

#reduce learning rate by fctor of 10 after 8 epochs
MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 GLOG_logtostderr=1 \
srun -n $NJOBS -p gpu --gres=gpu:3 --ntasks-per-node=3 \
$TOOLS/train_net.bin cifar10_quick_solver_lr1.prototxt cifar10_quick_iter_$N_FIRST_ITER.solverstate
