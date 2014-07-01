#!/usr/bin/env sh

TOOLS=`pwd`/../../build/tools

NJOBS=4
N_TASKS_PER_NODE=2
GRES_GPU=1

NJOBS=6
N_TASKS_PER_NODE=3
GRES_GPU=2

N_TRAIN_JOBS=`expr $NJOBS - 2`
N_FIRST_ITER=`expr 4000 / $N_TRAIN_JOBS`
N_SECOND_ITER=`expr 5000 / $N_TRAIN_JOBS`

cat cifar10_quick_solver.prototxt.template | sed "s/FIRST_ITER/$N_FIRST_ITER/g" > cifar10_quick_solver_pdcc.prototxt
cat cifar10_quick_solver_lr1.prototxt.template | sed "s/SECOND_ITER/$N_SECOND_ITER/g" > cifar10_quick_solver_lr1_pdcc.prototxt

MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 GLOG_logtostderr=1 \
srun -n $NJOBS -p gpu --gres=gpu:$GRES_GPU --ntasks-per-node=$N_TASKS_PER_NODE \
$TOOLS/train_net.bin cifar10_quick_solver_pdcc.prototxt # cifar10_quick_iter_init.solverstate

# exit 1

#reduce learning rate by fctor of 10 after 8 epochs
MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 GLOG_logtostderr=1 \
srun -n $NJOBS -p gpu --gres=gpu:$GRES_GPU --ntasks-per-node=$N_TASKS_PER_NODE \
$TOOLS/train_net.bin cifar10_quick_solver_lr1_pdcc.prototxt cifar10_quick_iter_$N_FIRST_ITER.solverstate
