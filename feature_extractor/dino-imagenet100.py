#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import torch
import torchvision

from datasets import load_dataset

import numpy as np

import os

from sl_finetuned_model import finetune_dino

def infer_features_labels(dino, data_loader, features_dir, labels_dir, device, args):

    dino.to(device)

    dino.eval()

    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for bidx, batch in enumerate(data_loader):

        images = batch["images"].to(device)

        if args.finetuned:
            features = dino(images).pooler_output
        else:
            features = dino(images)

        #np.save(f"origin/images_{bidx}",images.cpu().data)
        np.save(f"{features_dir}/features_{bidx}",features.cpu().data)
        np.save(f"{labels_dir}/labels_{bidx}",batch["labels"].cpu().data)

def merge_npy(features_dir, labels_dir, prefix, model_name, output_dir):

    # merge npy
    # Get sorted list of feature and label files
    feature_files = sorted([os.path.join(features_dir, f) for f in os.listdir(features_dir) if f.endswith('.npy')])
    label_files = sorted([os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.npy')])

    # Ensure matching feature and label files
    assert len(feature_files) == len(label_files), "Mismatch in number of feature and label files"

    def merged_array(files):
        arrays = [np.load(f) for f in files]
        return np.concatenate(arrays, axis=0)

    os.makedirs(f"{output_dir}/{model_name}", exist_ok=True)
    # Save the merged array to a new .npy file
    np.save(f"{output_dir}/{model_name}/{prefix['feature']}-{model_name}.npy", merged_array(feature_files))
    np.save(f"{output_dir}/{model_name}/{prefix['label']}-{model_name}.npy", merged_array(label_files))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DINO Inference on ImageNet100')

    parser.add_argument('--device', default='cuda', type=str, help='Device on which to run')
    parser.add_argument('--num-workers', default=8, type=int, help='Number of dataloader workers')

    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    
    parser.add_argument('--seed', default=42, type=int, help='random seed')

    parser.add_argument("--img_size", default=224, type=int, help="Resolution size")

    parser.add_argument("--model", default="dino_vitb16", type=str, help="Model name")

    parser.add_argument("--output_dir", default="output-features", type=str, help="Output directory")

    parser.add_argument("--finetuned", action='store_true', help='Finetuned model')

    parser.add_argument("--labeled_classes", default=50, type=int, help="Number of labeled classes")

    args = parser.parse_args()

    if args.seed != 0:
        torch.manual_seed(args.seed)

    # interpolation method = 3 (bicubic)
    interpolation = 3
    crop_pct = 0.875
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image_size = 224
    train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            torchvision.transforms.Resize(int(image_size / crop_pct), interpolation),
            torchvision.transforms.CenterCrop(image_size),
            torchvision.transforms.ColorJitter(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                        mean=torch.tensor(mean),
                        std=torch.tensor(std))
        ])

    # or load the separate splits if the dataset has train/validation/test splits
    train_dataset = load_dataset("clane9/imagenet-100", split="train")
    valid_dataset = load_dataset("clane9/imagenet-100", split="validation")
    #train_dataset = load_dataset("ilee0022/ImageNet100", split="train")
    #valid_dataset = load_dataset("ilee0022/ImageNet100", split="test")

    def trans_func(examples):
        images= [train_transforms(img) for img in examples["image"]]
        labels = torch.tensor(examples["label"])
        return {"images": images, "labels": labels}

    # create data loaders
    train_dataset.set_format("torch")
    train_dataset = train_dataset.with_transform(trans_func)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    data_loader = train_loader

    if (not args.finetuned):

        if args.model == "dinov2_vits14":

            dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        elif args.model == "dinov2_vitb14":

            dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

        elif args.model == "dino_vitb16":
            dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

        elif args.model == "dino_vitb8":
            dino = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')

        elif args.model == "dino_vits16":
            dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16') 

        else:
            raise ValueError("Model not supported")

        model_name = args.model.replace("_","-")

    else:

        # transfrom the dataset to have only 50 classes
        finetune_dataset = train_dataset.filter(lambda x: x["labels"] < 50)

        dino = finetune_dino(finetune_dataset, 100, model_name = args.model)

        model_name = args.model.replace("_","-")+"-sl"

    features_dir = f"{args.output_dir}/{args.model}_features"

    labels_dir = f"{args.output_dir}/{args.model}_labels"

    infer_features_labels(dino, data_loader, features_dir, labels_dir, args.device, args)

    merge_npy(features_dir, labels_dir, {"feature":"features", "label":"labels"}, model_name, args.output_dir)

    
    # create data loaders from the test set
    valid_dataset.set_format("torch")
    valid_dataset = valid_dataset.with_transform(trans_func)
    test_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size)
    data_loader = test_loader

    features_dir = f"{args.output_dir}/{args.model}_test_features"

    labels_dir = f"{args.output_dir}/{args.model}_test_labels"

    infer_features_labels(dino, data_loader, features_dir, labels_dir, args.device, args)

    merge_npy(features_dir, labels_dir, {"feature":"test_features", "label":"test_labels"}, model_name, args.output_dir)
    
# usage: python dino-cifar100.py 
