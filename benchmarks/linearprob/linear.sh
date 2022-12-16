python main_lincls.py \
  -a resnet50 \
  --dist-url 'tcp://localhost:10051' --multiprocessing-distributed --world-size 1 --rank 0 \
  --pretrained /path/to/pretrained/checkpoint \
  --lars \
  /path/to/imagenet/dataset
