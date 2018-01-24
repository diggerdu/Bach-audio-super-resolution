#!/usr/bin/env bash
#export CUDA_VISIBLE_DEVICES=2
expName=piano
selfPath=`realpath $0`
cd "$(git rev-parse --show-toplevel)"
mkdir -p checkpoints/$expName/
cp $selfPath checkpoints/$expName/
python train.py \
 --PathClean "/home/AAA/xiaver" \
 --PathNoise "/home/AAA/xiaver" \
 --snr 0 \
 --name $expName --model pix2pix --which_model_netG wide_resnet_3blocks \
 --ngf 32 \
 --which_direction AtoB --lambda_A 100 --no_lsgan --nThreads 6 \
 --nfft 1000 --hop 1000 --nFrames 16 --batchSize  64\
 --split_hop 0 \
 --niter 10000 --niter_decay 30 \
 --lr 1e-4 \
 --scale 2 \
 --layers 4 \
 --gpu_ids 0
# --input_nc 1 --output_nc 1 \
#  --serial_batches
