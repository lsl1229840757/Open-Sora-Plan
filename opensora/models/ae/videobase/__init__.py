"""
new code
"""
from .vqvae import (
    VQVAEConfiguration,
    VQVAEModel,
    VQVAETrainer,
    VQVAEDataset, VQVAEModelWrapper
)
from .causal_vqvae import (
    CausalVQVAEConfiguration,
    CausalVQVAEDataset,
    CausalVQVAETrainer,
    CausalVQVAEModel, CausalVQVAEModelWrapper
)
from .causal_vae import (
    CausalVAETrainer,
    CausalVAEConfiguration,
    CausalVAEDataset,
    CausalVAEModel
)
"""
old code
"""

videobase_ae_stride = {
    'CausalVQVAEModel_4x4x4': [4, 4, 4],
    'CausalVQVAEModel_4x8x8': [4, 8, 8],
    'VQVAEModel_4x4x4': [4, 4, 4],
    'OpenVQVAEModel_4x4x4': [4, 4, 4],
    'VQVAEModel_4x8x8': [4, 8, 8],
    'bair_stride4x2x2': [4, 2, 2],
    'ucf101_stride4x4x4': [4, 4, 4],
    'kinetics_stride4x4x4': [4, 4, 4],
    'kinetics_stride2x4x4': [2, 4, 4],
}

videobase_ae_channel = {
    'CausalVQVAEModel_4x4x4': 4,
    'CausalVQVAEModel_4x8x8': 4,
    'VQVAEModel_4x4x4': 4,
    'OpenVQVAEModel_4x4x4': 4,
    'VQVAEModel_4x8x8': 4,
    'bair_stride4x2x2': 256,
    'ucf101_stride4x4x4': 256,
    'kinetics_stride4x4x4': 256,
    'kinetics_stride2x4x4': 256,
}

videobase_ae = {
    'CausalVQVAEModel_4x4x4': CausalVQVAEModelWrapper,
    'CausalVQVAEModel_4x8x8': CausalVQVAEModelWrapper,
    'VQVAEModel_4x4x4': VQVAEModelWrapper,
    'VQVAEModel_4x8x8': VQVAEModelWrapper,
    "bair_stride4x2x2": VQVAEModelWrapper,
    "ucf101_stride4x4x4": VQVAEModelWrapper,
    "kinetics_stride4x4x4": VQVAEModelWrapper,
    "kinetics_stride2x4x4": VQVAEModelWrapper,
}
