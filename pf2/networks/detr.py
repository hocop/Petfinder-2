import numpy as np
import torch
import torch.nn as nn
import timm
import torchvision
from typing import Dict, Iterable, Callable, List


class PetDETR(nn.Module):
    def __init__(
        self,
        num_boxes,
        regression_margin,
    ):
        super().__init__()
        self.regression_margin = regression_margin
        self.num_boxes = num_boxes

        # Normalization layer
        self.norm_input = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Backbone
        model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet101_dc5', pretrained=True)
        # model.class_embed = nn.Identity()
        # model.bbox_embed = nn.Identity()
        self.backbone = FeatureExtractor(model, ['transformer.decoder.layers.1'])

        self.pred_embed = nn.Linear(256, num_boxes)
        self.out_layer = nn.Linear(100 * num_boxes + 14, 1)

    def forward(self, image, features):
        # Normalize input
        image = self.norm_input(image)

        # Extract features
        trans_feats = self.backbone(image)[0] # [100, B, 256]
        x = self.pred_embed(trans_feats)  # [100, B, feats/100]
        x = x.transpose(0, 1).reshape([-1, 100 * self.num_boxes]) # [B, feats]

        # Predict score
        x = torch.cat([x, features], dim=1)
        x = self.out_layer(x)

        # Scale
        out = torch.sigmoid(x[:, 0]) * (99 + self.regression_margin * 2) + 1 - self.regression_margin
        return out


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = [None for layer in self.layers]

        for name, layer in self.model.named_modules():
            if name in self.layers:
                layer_id = layers.index(name)
                hook = self._save_outputs_hook(layer_id)
                layer.register_forward_hook(hook)

    def _save_outputs_hook(self, layer_id: int) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
            # print(self.layers[layer_id], output.shape)
        return fn

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        _ = self.model(x)
        feats = self._features
        self._features = [None for layer in self.layers]
        return feats
