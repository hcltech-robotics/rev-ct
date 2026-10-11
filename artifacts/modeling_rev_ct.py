from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_rev_ct import RevConfig
from .head import RevHead
from .inference import create_image_encoder, encode_features

# Explicit re-export keeps this dependency in Transformers' local code cache on reload.
from .preprocessing import FastImageProcessor as FastImageProcessor


@dataclass
class RevOutput(ModelOutput):
    logits: torch.Tensor | None = None
    probabilities: torch.Tensor | None = None
    raw_probabilities: torch.Tensor | None = None
    decisions: torch.Tensor | None = None
    records: list[dict] | None = None


class RevModel(PreTrainedModel):
    config_class = RevConfig
    main_input_name = "pixel_values"
    base_model_prefix = ""

    def __init__(self, config: RevConfig):
        super().__init__(config)
        for name, spec in config.tensor_specs.items():
            parts = name.split(".")
            parent = self
            for part in parts[:-1]:
                if not hasattr(parent, part):
                    parent.add_module(part, nn.Module())
                parent = getattr(parent, part)
            parent.register_buffer(
                parts[-1], torch.zeros(spec["shape"], dtype=getattr(torch, spec["dtype"]))
            )
        self._inference = None
        if config.encoder_bundled:
            self.encoder = create_image_encoder(config.to_dict())
            self.encoder.requires_grad_(False)
        self.post_init()

    @property
    def dtype(self):
        if self.config.encoder_bundled:
            return next(self.encoder.parameters()).dtype
        return next(self.buffers()).dtype

    @property
    def device(self):
        return next(self.buffers()).device

    def _init_weights(self, module):
        if self.config.encoder_bundled:
            from timm.layers import RotaryEmbeddingCat

            if isinstance(module, RotaryEmbeddingCat) and module.pos_embed is not None:
                # Transformers rematerialises non-persistent meta buffers empty.
                # Generate on CPU, as COLIPRI does, even for device_map='cuda'.
                with torch.device("cpu"):
                    values = module._get_pos_embed_values(
                        module.feat_shape, device="cpu", dtype=torch.float32
                    )
                with torch.no_grad():
                    module.pos_embed.copy_(values)
                return
        super()._init_weights(module)

    def _rev_head(self) -> RevHead:
        arrays = {}
        for name, spec in self.config.tensor_specs.items():
            value = self.get_buffer(name)
            dtype = spec["dtype"]
            if value.dtype != getattr(torch, dtype):
                raise ValueError(
                    "Head tensor dtype changed; set precision on the encoder, not the Rev heads"
                )
            arrays[name] = value.detach().cpu().numpy()
        return RevHead(self.config.to_dict(), arrays)

    def forward(
        self,
        pixel_values=None,
        features=None,
        volume_names=None,
        head_mode=None,
        return_dict=True,
        **kwargs,
    ):
        if isinstance(pixel_values, dict):
            batch = pixel_values
            pixel_values, features = batch.get("pixel_values"), batch.get("features")
            volume_names = batch.get("volume_names", volume_names)
        if (pixel_values is None) == (features is None):
            raise ValueError("Provide exactly one of pixel_values or features")
        head = self._rev_head()
        device = next(self.buffers()).device
        if features is None:
            if self.config.encoder_bundled:
                parameter = next(self.encoder.parameters())
                features = encode_features(
                    self.encoder, pixel_values, parameter.device, parameter.dtype
                )
            else:
                from .inference import RevInference

                if self._inference is None:
                    self._inference = RevInference(head, device=str(device), precision="fp32")
                elif self._inference.device != device:
                    self._inference.device = device
                    self._inference.dtype = torch.float32
                    self._inference.model.to(device=device, dtype=self._inference.dtype)
                features = self._inference.encode(pixel_values)
        elif isinstance(features, torch.Tensor):
            features = features.detach().cpu().float().numpy()
        else:
            features = np.asarray(features)
        mode = head_mode or self.config.default_head_mode
        result = head.predict(features, mode=mode)
        names = (
            volume_names
            if volume_names is not None
            else [str(i) for i in range(len(result.scores))]
        )
        output = RevOutput(
            logits=torch.from_numpy(result.adjusted_logits).to(device),
            probabilities=torch.from_numpy(result.scores).to(device),
            raw_probabilities=torch.from_numpy(result.raw_probabilities).to(device),
            decisions=torch.from_numpy(result.decisions).to(device),
            records=head.records_from_output(result, names),
        )
        return output if return_dict else output.to_tuple()


RevConfig.register_for_auto_class("AutoConfig")
RevModel.register_for_auto_class("AutoModel")
