#!/usr/bin/env sh

TOOLS=`pwd`/../../build/tools

NJOBS=6
N_TRAIN_JOBS=`expr $NJOBS - 2`
N_MAX_ITER=`expr 450000 / $N_TRAIN_JOBS`

cat imagenet_solver.prototxt.template | sed "s/MAX_ITER/$N_MAX_ITER/g" > imagenet_solver_pdcc.prototxt

MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 GLOG_logtostderr=1 \
srun -n $NJOBS -p gpu --gres=gpu:4 --ntasks-per-node=6 \
$TOOLS/train_net.bin imagenet_solver_pdcc.prototxt

echo "Done."
