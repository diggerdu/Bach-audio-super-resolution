#!/usr/bin/env bash
#export CUDA_VISIBLE_DEVICES=2
expName=piano_gan
selfPath=`realpath $0`
cd "$(git rev-parse --show-toplevel)"
mkdir -p checkpoints/$expName/
cp $selfPath checkpoints/$expName/
python train.py \
 --PathClean "/home/alan/temp_audio" \
 --PathNoise "/home/alan/temp_audio" \
 --snr 0 \
 --name $expName --model pix2pix --which_model_netG wide_resnet_3blocks  \
 --which_model_netD seganDiscriminator \
 --gan_loss \
 --ngf 32 \
 --optimizer rmsprop \
 --which_direction AtoB --lambda_A 100 --nThreads 6 \
 --nfft 1024 --hop 1024 --nFrames 16 --batchSize  1\
 --split_hop 0 \
 --niter 10000 --niter_decay 30 \
 --lr 2e-4 \
 --scale 2 \
 --layers 4 \
 --gpu_ids 0
# --input_nc 1 --output_nc 1 \
#  --serial_batches
