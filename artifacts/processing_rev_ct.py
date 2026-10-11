from __future__ import annotations

import json
from pathlib import Path

from transformers import BatchFeature, ProcessorMixin


class RevProcessor(ProcessorMixin):
    attributes = []
    model_input_names = ["pixel_values"]

    def __init__(self, encoder=None, preprocessing="reference", **kwargs):
        if preprocessing not in ("reference", "fast"):
            raise ValueError("preprocessing must be reference or fast")
        self.encoder = encoder or {}
        self.preprocessing = preprocessing
        self._processor = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        path = Path(pretrained_model_name_or_path)
        if not path.is_dir():
            from huggingface_hub import hf_hub_download

            file = hf_hub_download(
                str(pretrained_model_name_or_path),
                "processor_config.json",
                revision=kwargs.get("revision"),
                token=kwargs.get("token"),
                local_files_only=kwargs.get("local_files_only", False),
            )
        else:
            file = path / "processor_config.json"
        config = json.loads(Path(file).read_text())
        return cls(
            encoder=config.get("encoder"),
            preprocessing=kwargs.get("preprocessing", config.get("preprocessing", "reference")),
        )

    def __call__(self, images3d=None, *, return_tensors="pt", **kwargs):
        if images3d is None or return_tensors != "pt":
            raise ValueError("Provide images3d=NIfTI paths and return_tensors='pt'")
        if self._processor is None:
            from importlib import import_module

            if self.preprocessing == "fast":
                from .preprocessing import FastImageProcessor

                self._processor = FastImageProcessor()
            else:
                self._processor = import_module("colipri").get_processor(image_only=True)
        batch = self._processor.to_images_batch(self._processor.process_images(images3d))
        return BatchFeature({"pixel_values": batch})

    def save_pretrained(self, save_directory, **kwargs):
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "processor_class": "RevProcessor",
            "encoder": self.encoder,
            "preprocessing": self.preprocessing,
            "auto_map": {"AutoProcessor": "processing_rev_ct.RevProcessor"},
        }
        (path / "processor_config.json").write_text(json.dumps(config, indent=2) + "\n")
        import shutil

        shutil.copyfile(__file__, path / "processing_rev_ct.py")
        shutil.copyfile(Path(__file__).with_name("preprocessing.py"), path / "preprocessing.py")
        return [str(path / "processor_config.json")]


RevProcessor.register_for_auto_class("AutoProcessor")
