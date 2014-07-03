#!/usr/bin/env sh

# Train image net with 4 GPUs on pdccmc1

TOOLS=`pwd`/../../build/tools

NJOBS=6
N_TASKS_PER_NODE=6
GRES_GPU=4
N_TRAIN_JOBS=`expr $NJOBS - 2`
N_MAX_ITER=`expr 450000 / $N_TRAIN_JOBS`

cat imagenet_solver.prototxt.template | sed "s/MAX_ITER/$N_MAX_ITER/g" > imagenet_solver_pdcc.prototxt

MV2_ENABLE_AFFINITY=0 MV2_USE_CUDA=1 GLOG_logtostderr=1 \
srun -n $NJOBS -p gpu --gres=gpu:$GRES_GPU --ntasks-per-node=$N_TASKS_PER_NODE \
$TOOLS/train_net.bin imagenet_solver_pdcc.prototxt

echo "Done."
