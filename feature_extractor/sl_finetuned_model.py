import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import AdamW
import torchvision
from torch.utils.data import DataLoader

from PIL import ImageFilter, ImageOps, Image
from torchvision import transforms

from peft import LoraConfig, get_peft_model, BOFTConfig
from transformers import AutoModel
from peft.peft_model import PeftModel
from peft.config import PeftConfig

from tqdm import tqdm
import copy

from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import vision_transformer as vits
from vision_transformer import DINOHead

import utils

import argparse
import os

torch.manual_seed(42)  # Set random seed for reproducibility

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        self.backbone = backbone

        self.head = head

    def forward(self, x):
        # Run the head forward on the concatenated features.
        pooler_output = self.backbone(x).pooler_output

        cls_output = self.head(pooler_output)

        return cls_output, pooler_output

def finetune_dino(train_set, num_classes, model_name="facebook/dino-vitb16"):
        
    interpolation = 3
    crop_pct = 0.875
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image_size = 224
    transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                        mean=torch.tensor(mean),
                        std=torch.tensor(std))
        ])

    batch_size = 128

    epochs = 10

    lora_model = load_model(num_classes, model_name)

    # Optimizer
    optimizer = AdamW(lora_model.parameters(), lr=1e-3)

    lora_model.to(device)

    print(train_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    # Training loop
    for epoch in range(epochs):  # Adjust epochs as needed
        for bidx, batch in tqdm(enumerate(train_loader)):

            images = batch["images"].to(device)

            output, pooler_output = lora_model(images)

            cls_loss = F.cross_entropy(output, batch["labels"].to(device))

            loss = cls_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return lora_model.backbone


def load_model(num_classes, model_name):

    if model_name == "dinov2_vitb14":
        model_name = "facebook/dinov2-base"

    if model_name == "dino_vitb16":
        model_name = "facebook/dino-vitb16"

    model = AutoModel.from_pretrained(model_name)


    peft_config = BOFTConfig(
        boft_block_size=4,
        boft_n_butterfly_factor=2,
        target_modules=["output.dense", "mlp.fc1", "mlp.fc2"],
        boft_dropout=0.1,
        bias="boft_only",
    )

    backbone = get_peft_model(model, peft_config)

    embed_dim = backbone.config.hidden_size

    lora_model = MultiCropWrapper(backbone, torch.nn.Linear(embed_dim, num_classes))

    lora_model.train()

    backbone.print_trainable_parameters()

    return lora_model
