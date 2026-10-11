from transformers import PretrainedConfig


class RevConfig(PretrainedConfig):
    model_type = "rev_ct"

    def __init__(
        self,
        labels=None,
        feature_dim=768,
        format_version="rev-safetensors-1",
        schema_version="rev-1.0",
        encoder=None,
        encoder_bundled=False,
        encoder_model_config=None,
        tensor_specs=None,
        calibration_input="raw_probability",
        default_head_mode="reference",
        **kwargs,
    ):
        kwargs.pop("num_labels", None)
        super().__init__(**kwargs)
        self.labels = labels or []
        self.feature_dim = feature_dim
        self.format_version = format_version
        self.schema_version = schema_version
        self.encoder = encoder or {}
        self.encoder_bundled = encoder_bundled
        self.encoder_model_config = encoder_model_config
        self.tensor_specs = tensor_specs or {}
        self.calibration_input = calibration_input
        self.default_head_mode = default_head_mode
        self.num_labels = len(self.labels)
        self.id2label = dict(enumerate(self.labels))
        self.label2id = {label: i for i, label in enumerate(self.labels)}
