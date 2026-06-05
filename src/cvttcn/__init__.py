"""CVTTCN: a hybrid Convolutional Vision Transformer + Temporal Convolutional
Network for EEG motor-imagery classification on the EEGMMIDB dataset."""

from cvttcn.config import Config, load_config

__version__ = "0.1.0"
__all__ = ["Config", "load_config", "__version__"]
