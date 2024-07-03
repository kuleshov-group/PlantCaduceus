"""Interpolation-based rotary embedding schemes.

"""
import math

import torch
from transformers.models.esm.modeling_esm import RotaryEmbedding


class RotaryEmbeddingPositionalInterpolation(RotaryEmbedding):
    """Position interpolation method.

    See ref: https://arxiv.org/abs/2306.15595
    """
    def __init__(self, dim, scaling_factor):
        """Scaling factor = L' / L, where is L' is the new context length."""
        super().__init__(dim)
        self.scaling_factor = scaling_factor

    def _update_cos_sin_tables(self, x, seq_dimension=2):
        seq_len = x.shape[seq_dimension]

        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = (torch.arange(x.shape[seq_dimension], device=x.device) / self.scaling_factor).type_as(self.inv_freq)

            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached


# noinspection PyMissingConstructor
class NTKAwareRotaryEmbedding(RotaryEmbedding):
    """Interpolation method by scaling the base number.

    See ref: https://arxiv.org/abs/2309.00071
    """

    def __init__(self, dim: int, scaling_factor: float):
        """Scaling factor = L' / L, where is L' is the new context length."""
        super().__init__(dim)
        updated_base = 10_000 * scaling_factor ** (dim / (dim - 2))

        inv_freq = 1.0 / (updated_base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None


class YaRNScaledRotaryEmbedding(RotaryEmbedding):
    """Interpolation method by scaling the base number.

    Implementation adapted from:
    https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    See ref: https://arxiv.org/abs/2309.00071
    """
    def __init__(
            self,
            dim,
            max_position_embeddings,
            original_max_position_embeddings,
            base=10_000,
            scale=1,
            extrapolation_factor=1,
            attn_factor=1,
            beta_fast=32,
            beta_slow=1,
    ):
        super().__init__(dim)
        self.dim = dim
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base
        self.scale = scale
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        self.mscale = 0.0
        self.yarn()

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()

        self.register_buffer("cos_cached", (emb.cos() * self.mscale)[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.mscale)[None, None, :, :].to(dtype), persistent=False)

    @staticmethod
    def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
        """Finds dim range bounds based on rotations."""
        def find_correction_dim(num_rotations):
            """Inverse dim formula to find dim based on number of rotations."""
            return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

        low = math.floor(find_correction_dim(low_rot))
        high = math.ceil(find_correction_dim(high_rot))
        return max(low, 0), min(high, dim - 1)  # Clamp values just in case

    @staticmethod
    def linear_ramp_mask(minimum, maximum, dim):
        """Defines ramp between interpolation and extrapolation."""
        if minimum == maximum:
            maximum += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=torch.float32) - minimum) / (maximum - minimum)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    @staticmethod
    def get_mscale(scale):
        """See Section 3.4 of YaRN paper."""
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    def yarn(self):
        """Initialize inv_freq and mscale."""
        # TODO(yair-schiff): Consider revising with corrected formula described here:
        #  https://github.com/jquesnelle/yarn/issues/24
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scale * pos_freqs)

        low, high = self.find_correction_range(
            self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings
        )
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (1 - self.linear_ramp_mask(low, high, self.dim // 2).float()) * self.extrapolation_factor
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

        self.register_buffer("inv_freq", inv_freq)
        # Get n-d magnitude scaling corrected for interpolation
        self.mscale = float(self.get_mscale(self.scale) * self.attn_factor)

    def _update_cos_sin_tables(self, x, seq_dimension=2):
        """Overrides RotaryEmbedding's method."""
        seq_len = x.shape[seq_dimension]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self.register_buffer(
                "cos_cached", (emb.cos() * self.mscale)[None, None, :, :].to(x.dtype), persistent=False
            )
            self.register_buffer(
                "sin_cached", (emb.sin() * self.mscale)[None, None, :, :].to(x.dtype), persistent=False
            )
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )
