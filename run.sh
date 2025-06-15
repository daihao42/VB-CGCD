#!/bin/bash 

python feature_extractor/dino-cifar100.py --finetuned --output_dir datasets/cifar100
python feature_extractor/dino-imagenet100.py --finetuned --output_dir datasets/imagenet100
python feature_extractor/dino-tinyimagenet.py --finetuned --output_dir datasets/tinyimagenet
python feature_extractor/dino-cub200.py --finetuned --output_dir datasets/cub200

python main.py --base 50 --increment 10 --pretrained_model_name dino-vitb16-sl --data_dir datasets/cifar100 --trail_name mix_increment_mngmm_dinovb16_sl_cifar_100
python main.py --base 100 --increment 20 --pretrained_model_name dino-vitb16-sl --dataset tinyimagenet --data_dir datasets/tinyimagenet --num_classes 200 --trail_name mix_increment_mngmm_dinovb16_sl_tiny_imagenet
python main.py --base 50 --increment 10 --pretrained_model_name dino-vitb16-sl --dataset imagenet100 --data_dir datasets/imagenet100 --trail_name mix_increment_mngmm_dinovb16_sl_imagenet100
python main.py --base 100 --increment 20 --pretrained_model_name dino-vitb16-sl --dataset cub200 --data_dir datasets/cub200 --num_classes 200 --trail_name mix_increment_mngmm_dinovb16_sl_cub200
