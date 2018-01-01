#!/usr/bin/env bash
#export CUDA_VISIBLE_DEVICES=2
expName=dict
selfPath=`realpath $0`
cd "$(git rev-parse --show-toplevel)"
mkdir -p checkpoints/$expName/
cp $selfPath checkpoints/$expName/
python train.py \
 --PathClean "/home/alan/Documents/once" \
 --PathNoise "/home/alan/Documents/once" \
 --snr 0 \
 --name $expName --model pix2pix --which_model_netG wide_resnet_3blocks \
 --ngf 32 \
 --which_direction AtoB --lambda_A 100 --no_lsgan --nThreads 6 \
 --nfft 1024 --hop 512 --nFrames 8 --batchSize  1\
 --split_hop 0 \
 --niter 10000 --niter_decay 30 \
 --lr 1e-4 \
 --scale 2 \
 --layers 4 \
 --gpu_ids 0 \
 --serial_batches \
 --nmfcc 13
# --input_nc 1 --output_nc 1 \
