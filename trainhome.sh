# !/bin/bash

export CUDA_VISIBLE_DEVICES=3
python -m domainbed.scripts.train\
       --data_dir /l/users/zhongyi.han/dataset\
       --output_dir ./logs\
       --algorithm MetricSoftmaxAlignPatch\
       --dataset OfficeHome\
       --hparams "{\"clip_backbone\": \"ViT-B/16\"}"\
       --test_envs 0
# export CUDA_VISIBLE_DEVICES=2
# python -m domainbed.scripts.train\
#        --data_dir /l/users/zhongyi.han/dataset\
#        --output_dir ./logs\
#        --algorithm MetricSoftmaxAlignPatch\
#        --dataset OfficeHome\
#        --hparams "{\"clip_backbone\": \"ViT-B/16\"}"\
#        --test_envs 1
# export CUDA_VISIBLE_DEVICES=2
# python -m domainbed.scripts.train\
#        --data_dir /l/users/zhongyi.han/dataset\
#        --output_dir ./logs\
#        --algorithm MetricSoftmaxAlignPatch\
#        --dataset OfficeHome\
#        --hparams "{\"clip_backbone\": \"ViT-B/16\"}"\
#        --test_envs 2
# export CUDA_VISIBLE_DEVICES=2
# python -m domainbed.scripts.train\
#        --data_dir /l/users/zhongyi.han/dataset\
#        --output_dir ./logs\
#        --algorithm MetricSoftmaxAlignPatch\
#        --dataset OfficeHome\
#        --hparams "{\"clip_backbone\": \"ViT-B/16\"}"\
#        --test_envs 3