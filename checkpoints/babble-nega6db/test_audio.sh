export CUDA_VISIBLE_DEVICES=2
expName=nega6db
selfPath=`realpath $0`
cd "$(git rev-parse --show-toplevel)"
cp $selfPath checkpoints/$expName/

python test.py \
 --PathClean "/home/diggerdu/dataset/men/clean" \
 --PathNoise "/home/diggerdu/dataset/men/noise" \
 --snr -6 \
 --name $expName --model pix2pix --which_model_netG wide_resnet_3blocks \
 --ngf 32 \
 --input_nc 1 --output_nc 1  \
 --which_direction AtoB --nThreads 1 \
 --nfft 1024 --hop 512 --nFrames 128 \
 --gpu_ids 0 --batchSize 1  --how_many 6 \
 --split_hop 0 \

mv *.wav checkpoints/$expName/

