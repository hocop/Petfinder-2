import numpy as np
import torch
import torch.nn as nn
import timm
import torchvision


class SWIN(nn.Module):
    def __init__(self, model_name, num_image_neurons, dropout, attn_drop):
        super().__init__()

        # Normalization layer
        self.norm_input = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Backbone
        self.pet_net = timm.create_model(model_name, pretrained=True, in_chans=3, attn_drop=attn_drop)

        # Freeze first layers
        self.pet_net.patch_embed.requires_grad_(False)
        self.pet_net.pos_drop.requires_grad_(False)
        self.pet_net.layers[1].requires_grad_(False)
        self.pet_net.layers[2].requires_grad_(False)

        # Replace head
        self.pet_net.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                self.pet_net.head.in_features,
                num_image_neurons
            ),
            nn.LayerNorm(num_image_neurons),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

        self.out_layer = nn.Linear(num_image_neurons + 14, 1)

    def forward(self, image, features):
        # Normalize input
        image = self.norm_input(image)

        # Extract features
        x = self.pet_net(image)

        # Predict score
        x = torch.cat([x, features], dim=1)
        x = self.out_layer(x)
        return x