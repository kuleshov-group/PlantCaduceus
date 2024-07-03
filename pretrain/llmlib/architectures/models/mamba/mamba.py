import torch
import torch.nn.functional as F
import lightning as L
import math
import wandb
import einops
import torchvision

from .base import BaseLm
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import decode

LOG2 = math.log(2)

def get_model(args):
    if args.model == "MambaSubPixelLm":
        return MambaSubPixelLm(args)
    else:
        raise ValueError(f"Model {args.model} not implemented")

class MambaSubPixelLm(BaseLm):
    def create_model(self):
        return MambaLMHeadModel(
            self.args.d_model,
            self.args.n_layer,
            self.args.vocab_size,
        )

    def forward(self, x):
        return self.model(x.view(x.shape[0], -1))

    # sampling helper functions
    @torch.inference_mode()
    def sample(self, num_samples):
        x = torch.full((num_samples, 1), 256, dtype=torch.long, device=self.device)
        output = decode(x, self.model, 32 * 32 * 3 + 1, top_k=0)
        return (
            einops.rearrange(
                output.sequences[:, 1:],
                "b (h w c) -> b c h w",
                h=32,
                w=32,
                c=3,
            ).float()
            / 255
        )

    def sample_wandb_grid(self, num_samples):
        samples = self.sample(num_samples)
        image_grid = torchvision.utils.make_grid(samples)
        images = wandb.Image(image_grid)
        return images
