import numpy as np
import torch
import torch.nn as nn
import timm
import torchvision
from einops import rearrange


class SWIN(nn.Module):
    def __init__(
        self,
        model_name,
        num_image_neurons,
        dropout_1,
        dropout_2,
        attn_drop,
        attn_drop_final,
        freeze_layers,
        regression_margin_bot,
        regression_margin_top,
    ):
        super().__init__()
        self.regression_margin_bot = regression_margin_bot
        self.regression_margin_top = regression_margin_top

        # Normalization layer
        self.norm_input = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Backbone
        self.pet_net = timm.create_model(model_name, pretrained=True, in_chans=3, attn_drop=attn_drop)

        # Freeze first layers
        if freeze_layers >= 1:
            self.pet_net.patch_embed.requires_grad_(False)
            self.pet_net.pos_drop.requires_grad_(False)
        if freeze_layers >= 2:
            self.pet_net.layers[1].requires_grad_(False)
        if freeze_layers >= 3:
            self.pet_net.layers[2].requires_grad_(False)
        if freeze_layers >= 4:
            self.pet_net.layers[3].requires_grad_(False)

        # Replace pooling
        self.pet_net.avgpool = nn.Identity()
        self.att_pool = AttPool(
            self.pet_net.head.in_features,
            attn_drop=attn_drop_final,
        )

        # Replace head
        self.image_dense_layer = nn.Sequential(
            nn.Linear(
                self.pet_net.head.in_features,
                num_image_neurons
            ),
            nn.LayerNorm(num_image_neurons),
            nn.ReLU(True),
        )
        self.pet_net.head = nn.Identity()

        self.dropout_1 = nn.Dropout(dropout_1)
        self.dropout_2 = nn.Dropout(dropout_2)

        self.out_layer = nn.Linear(num_image_neurons + 14, 1)

        # Initialize weights
        # nn.init.xavier_uniform_(self.image_dense_layer[0].weight)
        # nn.init.xavier_uniform_(self.out_layer.weight)

    def forward(self, image, features, freeze_backend=torch.tensor(False)):
        # Normalize input
        image = self.norm_input(image)

        # Extract features
        x = self.pet_net(image)
        if freeze_backend:
            x = x.detach()

        # Attention over image
        # x = rearrange(x, 'b (c l) -> b c l', c=self.att_pool.in_features)
        x = x.reshape([x.shape[0], self.att_pool.in_features, -1]) # [B, C, L]
        x = self.att_pool(x) # [B, C]
        image_features = x

        # Feedforward layer
        x = self.dropout_1(x)
        x = self.image_dense_layer(x)
        x = self.dropout_2(x)

        # Predict score
        x = torch.cat([x, features], dim=1)
        x = self.out_layer(x)

        # Scale
        a = self.regression_margin_bot
        b = self.regression_margin_top
        out = torch.sigmoid(x) * (99 + a + b) + 1 - a
        return out, image_features



class AttPool(nn.Module):
    def __init__(
        self,
        in_features,
        attn_drop=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.to_key = nn.Linear(in_features, 1, bias=False)
        self.attn_drop = attn_drop

    def forward(self, x):
        '''
        x: tensor with shape [B, C, L]
        '''
        key = self.to_key(x.transpose(1, 2))[:, :, 0] # [B, L]
        key = torch.softmax(key, dim=1)
        # Word dropout
        if self.training and self.attn_drop != 0:
            mask = torch.rand_like(key)
            mask = (mask < self.attn_drop).to(torch.float32)
            key = key * mask
            key = key / (key.sum(dim=1, keepdim=True) + 1e-8)
        # Attention
        c = torch.einsum('bcl,bl->bc', x, key)
        return c # [B, C]



class SWIN_simple(nn.Module):
    def __init__(
        self,
        model_name,
        dropout_1,
        attn_drop,
        freeze_layers,
        regression_margin_bot,
        regression_margin_top,
    ):
        super().__init__()
        self.regression_margin_bot = regression_margin_bot
        self.regression_margin_top = regression_margin_top

        # Normalization layer
        self.norm_input = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Backbone
        self.pet_net = timm.create_model(model_name, pretrained=True, in_chans=3, attn_drop=attn_drop)

        # Freeze first layers
        if freeze_layers >= 1:
            self.pet_net.patch_embed.requires_grad_(False)
            self.pet_net.pos_drop.requires_grad_(False)
        if freeze_layers >= 2:
            self.pet_net.layers[1].requires_grad_(False)
        if freeze_layers >= 3:
            self.pet_net.layers[2].requires_grad_(False)
        if freeze_layers >= 4:
            self.pet_net.layers[3].requires_grad_(False)

        # Replace head
        self.pet_net.head = nn.Sequential(
            nn.Dropout(dropout_1),
            nn.Linear(self.pet_net.head.in_features, 1),
        )
        

    def forward(self, image, features, freeze_backend=torch.tensor(False)):
        # Normalize input
        image = self.norm_input(image)

        # Extract features
        x = self.pet_net(image)

        # Scale
        a = self.regression_margin_bot
        b = self.regression_margin_top
        out = torch.sigmoid(x) * (99 + a + b) + 1 - a
        return out, None
