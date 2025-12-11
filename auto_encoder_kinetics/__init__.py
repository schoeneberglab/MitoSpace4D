"""
MitoSpace Kinetics Autoencoder Library

This package provides autoencoder models, training utilities, and data processing
tools for MitoSpace 4D microscopy data.

Main Components:
- Models: MitoSpace3DAutoencoder, MitoSpace3DEncoder, MitoSpace3DDecoder
- Training: AutoEncoderRunner
- Dataset: MitoSpaceAutoEncoderDataset
- Loss: ReconstructionLoss
- Utilities: AEUtil
"""

__version__ = "1.0.0"

# Import main classes and functions
from .autoencoder_models import (
    MitoSpace3DAutoencoder as MitoSpace3DAutoencoderBase,
    MitoSpace3DEncoder as MitoSpace3DEncoderBase,
    MitoSpace3DDecoder as MitoSpace3DDecoderBase,
)

from .autoencoder_models_resnet import (
    MitoSpace3DAutoencoder,
    MitoSpace3DEncoder,
    MitoSpace3DDecoder,
)

from .autoencoder_runner import AutoEncoderRunner

from .autoencoder_dataset import (
    MitoSpaceAutoEncoderDataset,
    NormalizeChannelsByPath,
)

from .ae_loss import ReconstructionLoss

from .ae_util import AEUtil

# Convenience exports
__all__ = [
    # Models (ResNet version - recommended)
    "MitoSpace3DAutoencoder",
    "MitoSpace3DEncoder",
    "MitoSpace3DDecoder",
    # Models (Base version)
    "MitoSpace3DAutoencoderBase",
    "MitoSpace3DEncoderBase",
    "MitoSpace3DDecoderBase",
    # Training
    "AutoEncoderRunner",
    # Dataset
    "MitoSpaceAutoEncoderDataset",
    "NormalizeChannelsByPath",
    # Loss
    "ReconstructionLoss",
    # Utilities
    "AEUtil",
]

