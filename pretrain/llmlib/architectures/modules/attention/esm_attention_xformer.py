"""xFormer-based attention module.

"""

from typing import Optional, Tuple

import torch
from transformers.models.esm.modeling_esm import EsmSelfAttention
from xformers.components.attention import ScaledDotProduct, LocalAttention
import xformers.ops as xops

from .strategy_enums import AttentionStrategy


class EsmSelfAttention_xFormer(EsmSelfAttention):
    """xFormer-based attention module.

    Supported methods:
        - xFormer: (vanilla) xFormer attention
        - xFormer_efficient: more memory efficient implementation of xFormer attention
        - xFormer_local: xFormer with local attention
    """
    def __init__(self, config, position_embedding_type, attention_strategy):
        super().__init__(config, position_embedding_type)
        self.attention_strategy = attention_strategy
        if attention_strategy == AttentionStrategy.XFORMER:
            self.xformer_attn = ScaledDotProduct(dropout=config.attention_probs_dropout_prob)
        elif attention_strategy == AttentionStrategy.XFORMER_LOCAL:
            self.xformer_attn = LocalAttention(
                causal=False,
                dropout=config.attention_probs_dropout_prob,
                window_size=config.local_attention_window_size,
            )
        elif attention_strategy == AttentionStrategy.XFORMER_EFFICIENT:
            self.xformer_attn = None
        else:
            raise NotImplementedError(f"Attention strategy {attention_strategy} not implemented.")

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        """Override self-attention forward pass."""
        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        if self.position_embedding_type == "rotary":
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)

        if self.attention_strategy == AttentionStrategy.XFORMER_LOCAL:
            bsz, n_head, seq_len, head_dim = query_layer.shape
            query_layer = query_layer.reshape(bsz * n_head, seq_len, head_dim)
            key_layer = key_layer.reshape(bsz * n_head, seq_len, head_dim)
            value_layer = value_layer.reshape(bsz * n_head, seq_len, head_dim)
            context_layer = self.xformer_attn(query_layer, key_layer, value_layer)
            context_layer = context_layer.view(bsz, n_head, seq_len, head_dim)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # Reshape: (bsz, seq_len, n_heads. head_dim)
        elif self.attention_strategy == AttentionStrategy.XFORMER_EFFICIENT:
            query_layer = query_layer.permute(0, 2, 1, 3)
            key_layer = key_layer.permute(0, 2, 1, 3)
            value_layer = value_layer.permute(0, 2, 1, 3)
            # Shape: (bsz, seq_len, n_heads. head_dim)
            # Original: After rotary_embeddings, tensor becomes float32.
            # context_layer = xops.memory_efficient_attention(query_layer, key_layer, value_layer.type(query_layer.dtype))
            # Use Float16 for attention, it reduces
            context_layer = xops.memory_efficient_attention(query_layer.type(value_layer.dtype), key_layer.type(value_layer.dtype), value_layer)
        else:
            context_layer = self.xformer_attn(query_layer, key_layer, value_layer, att_mask=attention_mask)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # Reshape: (bsz, seq_len, n_heads. head_dim)

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # Reshape: (bsz, seq_len, n_heads*head_dim)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer,)
        return outputs
