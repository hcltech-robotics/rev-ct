"""Rev CT inference and installed Transformers architecture registration."""

from importlib.util import find_spec

from .head import RevHead

__version__ = "0.3.2"
__all__ = ["RevHead", "register_auto_classes"]


def register_auto_classes():
    """Enable AutoModel/AutoProcessor for code-free decision-model checkpoints."""
    from transformers import AutoConfig, AutoModel, AutoProcessor

    from .configuration_rev_ct import LegacyRevConfig, RevConfig
    from .modeling_rev_ct import LegacyRevModel, RevModel
    from .processing_rev_ct import RevProcessor

    for config, model in [(RevConfig, RevModel), (LegacyRevConfig, LegacyRevModel)]:
        AutoConfig.register(config.model_type, config, exist_ok=True)
        AutoModel.register(config, model, exist_ok=True)
        AutoProcessor.register(config, RevProcessor, exist_ok=True)


if find_spec("transformers") is not None and find_spec("torch") is not None:
    register_auto_classes()
