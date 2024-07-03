"""Utilities for model manipulation.

"""

from .attention.esm_attention_xformer import EsmSelfAttention_xFormer
from .attention.rotary_embeddings import (
    NTKAwareRotaryEmbedding, RotaryEmbeddingPositionalInterpolation, YaRNScaledRotaryEmbedding
)

from .attention.strategy_enums  import PositionalEmbeddingStrategy


def replace_attention_modules(model, config, attention_strategy, local_attention_window_size=None):
    """Replace attention modules (inplace).

    Recursively search modules in a model and replace attention mechanisms.
    """
    config.local_attention_window_size = local_attention_window_size
    for name, module in model.named_children():
        if name == "self":
            new_module = EsmSelfAttention_xFormer(
                config,
                module.position_embedding_type,
                attention_strategy=attention_strategy,
            )
            for param_name, param in module.named_parameters():
                for param_name_new, param_new in new_module.named_parameters():
                    if param_name == param_name_new:
                        param_new.data.copy_(param.data)
            setattr(model, name, new_module)
        else:
            replace_attention_modules(
                module, config, attention_strategy, local_attention_window_size=local_attention_window_size
            )


def replace_embedding_module(
        model, positional_embedding_strategy, max_position_embeddings, scaling_factor, layers_to_replace
):
    """Replace embedding module (inplace)."""
    for layer in layers_to_replace:
        assert layer < len(model.esm.encoder.layer), f"Layer index must be in [0, {len(model.esm.encoder.layer) - 1}]."
    dim = int(model.config.hidden_size / model.config.num_attention_heads)
    for layer in layers_to_replace:
        if positional_embedding_strategy == PositionalEmbeddingStrategy.INTERPOLATE:
            new_module = RotaryEmbeddingPositionalInterpolation(dim=dim, scaling_factor=scaling_factor)
        elif positional_embedding_strategy == PositionalEmbeddingStrategy.NTK:
            new_module = NTKAwareRotaryEmbedding(
                dim=dim,
                scaling_factor=scaling_factor,
            )
        elif positional_embedding_strategy == PositionalEmbeddingStrategy.YARN:
            new_module = YaRNScaledRotaryEmbedding(
                dim=dim,
                max_position_embeddings=max_position_embeddings,
                original_max_position_embeddings=model.config.max_position_embeddings,
                base=10_000,
                scale=scaling_factor,
                # TODO: Default params from YaRN, which were tuned to Llama-models. Consider changing.
                extrapolation_factor=1,
                attn_factor=1,
                beta_fast=32,
                beta_slow=1,
            )
        else:
            raise NotImplementedError(f"Positional embedding strategy {positional_embedding_strategy} not implemented.")
        model.esm.encoder.layer[layer].attention.self.rotary_embeddings = new_module
