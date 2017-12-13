#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
expName=allnoise-0db-spec
selfPath=`realpath $0`
cd "$(git rev-parse --show-toplevel)"
mkdir -p checkpoints/$expName/
cp $selfPath checkpoints/$expName/
python train.py \
 --PathClean "/home/diggerdu/dataset/Large/clean" \
 --PathNoise "/home/diggerdu/dataset/Large/allnoise" \
 --snr 0 \
 --name $expName --model pix2pix --which_model_netG wide_resnet_3blocks \
 --specLoss  --specLossStart 256 --specLossRatio 0.3 \
 --ngf 32 \
 --which_direction AtoB --lambda_A 100 --no_lsgan --nThreads 6 \
 --nfft 1024 --hop 512 --nFrames 128 --batchSize  7\
 --split_hop 0 \
 --niter 100000000000000000000000000000000000 --niter_decay 30 \
 --lr 0.00001 \
 --gpu_ids 0 \
 --continue_train
#  --serial_batches
