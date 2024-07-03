import torch
import torch.nn.functional as F
import lightning as L
import wandb
import einops
import torchvision

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.generation import decode
from ssmim.constants import LOG2


class BaseLm(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.vocab_size = args.vocab_size
        self.model = self.create_model()

        # logging metrics
        self.train_loss = 0
        self.train_n = 0
        self.valid_loss = 0
        self.valid_n = 0
        self.test_loss = 0
        self.test_n = 0

    def create_model(self):
        raise NotImplementedError

    def forward(self, x):
        return self.model(x)

    def nll(self, batch):
        # batch: batch x time x ... anything else
        inputs = batch[:, :-1]
        labels = batch[:, 1:]
        output = self.model(inputs)
        nll = F.cross_entropy(
            output.logits.view(-1, self.vocab_size),
            labels.reshape(-1),
            reduction="none",
        )
        return nll

    def bpd(self, batch):
        return self.nll(batch) / LOG2

    # loop functions
    def training_step(self, batch, batch_idx):
        bpd = self.bpd(batch)
        loss = bpd.mean()
        self.log("train/bpd", loss, on_step=True,on_epoch=True, prog_bar=True, sync_dist=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        bpd = self.bpd(batch)
        self.log("valid/bpd", bpd.mean(), on_step=False, on_epoch=True, sync_dist=True, logger=True)

    def on_validation_epoch_end(self):
        samples = self.sample_wandb_grid(16)
        self.logger.experiment.log(
            {
                "samples": samples,
            },
            step = self.global_step,
        )

    def test_step(self, batch, batch_idx):
        bpd = self.bpd(batch)
        self.log("test/bpd", bpd.mean(), on_step=False, on_epoch=True, sync_dist=True, logger=True)

    def on_test_epoch_end(self):
        samples = self.sample_wandb_grid(16)
        self.logger.experiment.log(
            {
                "samples": samples,
            },
            step = self.global_step+1,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.args.patience,
            factor=self.args.factor,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid/bpd",
            },
        }

    # sampling helper functions
    @torch.inference_mode()
    def sample(self, num_samples):
        raise NotImplementedError

    def sample_wandb_grid(self, num_samples):
        samples = self.sample(num_samples)
        image_grid = torchvision.utils.make_grid(samples)
        images = wandb.Image(image_grid)
        return images
