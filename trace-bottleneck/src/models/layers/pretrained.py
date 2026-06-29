from env import CACHE
import os
from typing import Dict
import torch
from torch import nn
from torch import nn
from torchvision.models.resnet import resnet50
from transformers import AutoConfig, AutoModel, SwinModel, ViTModel, BertModel

class MLPProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class LinearProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)

    def forward(self, x):
        return self.projection(x)

class ResNet50(nn.Module):
    def __init__(self, name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.resnet = resnet50(pretrained=True)
        else:
            # TODO: add vision models if needed
            raise NotImplementedError(f"Not support training from scratch : {name}")

        self.out_dim = 2048
        del self.resnet.fc
        self.resnet = nn.SyncBatchNorm.convert_sync_batchnorm(self.resnet)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x


class InputImgEncoder(torch.nn.Module):
    """
    Initialize the input image encoder.

    Attributes:
        original_model: The original model to extract features from.
    """
    def __init__(self, original_model: torch.nn.Module):
        super(InputImgEncoder, self).__init__()
        self.features = torch.nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the input image encoder.

        Args:
            x: The input tensor.

        Returns:
            torch.Tensor: The output tensor from the last layer of the model.
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x
    

class HuggingfaceImageEncoder(nn.Module):
    def __init__(
        self,
        name: str = "google/vit-base-patch16-224",
        pretrained: bool = True,
        gradient_checkpointing: bool = False,
        cache_dir: str = str(CACHE/ "huggingface"),
        model_type: str = "vit",
        local_files_only: bool = False,
    ):
        super().__init__()
        self.model_type = model_type
        if pretrained:
            if self.model_type == "swin":
                self.image_encoder = SwinModel.from_pretrained(name)
            else:
                self.image_encoder = AutoModel.from_pretrained(
                    name, add_pooling_layer=False, cache_dir=cache_dir, local_files_only=local_files_only
                )
        else:
            # initializing with a config file does not load the weights associated with the model
            model_config = AutoConfig.from_pretrained(name, cache_dir=cache_dir, local_files_only=local_files_only)
            if type(model_config).__name__ == "ViTConfig":
                self.image_encoder = ViTModel(model_config, add_pooling_layer=False)
            else:
                # TODO: add vision models if needed
                raise NotImplementedError(f"Not support training from scratch : {type(model_config).__name__}")

        if gradient_checkpointing and self.image_encoder.supports_gradient_checkpointing:
            self.image_encoder.gradient_checkpointing_enable()

        self.out_dim = self.image_encoder.config.hidden_size

    def forward(self, image):
        if self.model_type == "vit":
            output = self.image_encoder(pixel_values=image, interpolate_pos_encoding=True)
        elif self.model_type == "swin":
            output = self.image_encoder(pixel_values=image)
        return output["last_hidden_state"]  # (batch, seq_len, hidden_size)
    
class HuggingfaceTextEncoder(nn.Module):
    def __init__(
        self,
        name: str = "bert-base-uncased",
        vocab_size: int = None,
        pretrained: bool = True,
        gradient_checkpointing: bool = False,
        cache_dir: str = str(CACHE/ "huggingface"),
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ):
        super().__init__()
        if pretrained:
            self.text_encoder = AutoModel.from_pretrained(
                name,
                # vocab_size=vocab_size,
                ignore_mismatched_sizes=True,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
        else:
            # initializing with a config file does not load the weights associated with the model
            model_config = AutoConfig.from_pretrained(
                name,
                # vocab_size=vocab_size,
                ignore_mismatched_sizes=True,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
            if type(model_config).__name__ == "BertConfig":
                self.text_encoder = BertModel(model_config)
            else:
                # TODO: add text models if needed
                raise NotImplementedError(f"Not support training from scratch : {type(model_config).__name__}")

        if gradient_checkpointing and self.text_encoder.supports_gradient_checkpointing:
            self.text_encoder.gradient_checkpointing_enable()

        self.out_dim = self.text_encoder.config.hidden_size

    def forward(self, x):
        output = self.text_encoder(**x)
        return output["last_hidden_state"]  # (batch, seq_len, hidden_size)
