#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
expName=0db-128
selfPath=`realpath $0`
cd "$(git rev-parse --show-toplevel)"
mkdir -p checkpoints/$expName/
cp $selfPath checkpoints/$expName/
python train.py \
 --PathClean "/home/diggerdu/dataset/men/clean" \
 --PathNoise "/home/diggerdu/dataset/men/noise" \
 --snr 0 \
 --name $expName --model pix2pix --which_model_netG wide_resnet_3blocks \
 --ngf 32 \
 --which_direction AtoB --lambda_A 100 --no_lsgan --nThreads 6 \
 --input_nc 1 --output_nc 1 \
 --nfft 256 --hop 128 --nFrames 128 --batchSize  24\
 --split_hop 0 \
 --niter 100000000000000000000000000000000000 --niter_decay 30 \
 --lr 0.0001 \
 --gpu_ids 0 \
 --continue_train
#  --serial_batches
