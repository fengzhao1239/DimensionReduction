import torch
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
from einops import rearrange
from torch import nn
from time import time
import numpy as np
import matplotlib.pyplot as plt
import wandb
import random
    

class LOSS(nn.Module):
    def __init__(self, beta=1e-4, eps=1e-12, channel_weights=None):
        """
        Functional Autoencoder Loss using channel-wise Relative L2 Norm.
        
        Args:
            beta (float): Penalty coefficient for latent representation z.
            eps (float): Small value to prevent division by zero.
            channel_weights (list or None): Weights for each output channel (C).
        """
        super().__init__()
        self.beta = beta
        self.eps = eps
        
        self.register_buffer(
            "channel_weights",
            None if channel_weights is None else torch.tensor(channel_weights, dtype=torch.float32)
        )

    def forward(self, u_pred, u_target, latents):
        """
        Args:
            u_pred: [B, N, C] or [B, H, W, C] or higher dims
            u_target: Same shape as u_pred
            latents: Latent vector z [B, latent_dim]
        """
        if u_pred.dim() > 3:
            # Reshape all spatial dimensions to (B, N, C)
            u_pred = rearrange(u_pred, 'b ... c -> b (...) c')
            u_target = rearrange(u_target, 'b ... c -> b (...) c')

        diff_norm = torch.linalg.norm(u_pred - u_target, ord=2, dim=1)
        target_norm = torch.linalg.norm(u_target, ord=2, dim=1)
        rel_error = diff_norm / (target_norm + self.eps)
        
        if self.channel_weights is None:
            reconstruction_loss = rel_error.mean()
        else:
            w = self.channel_weights
            w = w / (w.sum() + self.eps)
            reconstruction_loss = (rel_error * w[None, :]).sum(dim=-1).mean()
        
        regularization_loss = self.beta * torch.mean(torch.sum(latents**2, dim=-1))
        total_loss = reconstruction_loss + regularization_loss
        return total_loss, reconstruction_loss, regularization_loss


class FAETrainer(L.LightningModule):
    def __init__(
        self,
        model,
        training_config,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.automatic_optimization = False
        self.loss_fn = LOSS(
            beta=training_config.get('beta', 1e-4),
            eps=training_config.get('eps', 1e-12),
            channel_weights=training_config.get('channel_weights', None)
        )
        for k, v in training_config.items():
            setattr(self, k, v)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        x_enc = batch['coords_enc']
        u_enc = batch['foi_enc']
        x_dec = batch['coords_dec']
        u_dec = batch['foi_dec']
        
        u_pred, latents, aux_loss = self.model(x_enc, u_enc, query_coords=x_dec, return_latents=True)
        ori_loss, reconstruction_loss, regularization_loss = self.loss_fn(u_pred, u_dec, latents)
        
        # aux_loss contains VQ loss (if configured)
        loss = ori_loss + aux_loss if aux_loss is not None else ori_loss
        
        if self.use_lipschitz:
            lipschitz_loss = self.model.decoder.lipschitz_loss()
            self.log(f"lipschitz_loss", lipschitz_loss.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            loss = loss + self.lipschitz_coeff * lipschitz_loss
        
        opt.zero_grad()
        self.manual_backward(loss)
        # if self.global_rank == 0 and batch_idx == 0:  
        #     unused = [n for n,p in self.named_parameters() if p.requires_grad and p.grad is None]
        #     print("Unused:", len(unused))
        #     print("\n".join(unused[:50]))
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_val)
        opt.step()
        sch.step()
        
        # Log losses
        self.log(f"train_loss", ori_loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if aux_loss is not None:
            self.log(f"train_aux_loss", aux_loss.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        coord, field = batch
        recon, latents, aux_loss = self.model(coord, field, return_latents=True)
        ori_loss, reconstruction_loss, regularization_loss = self.loss_fn(recon, field, latents)
        
        # aux_loss contains VQ loss (if configured)
        loss = ori_loss + aux_loss if aux_loss is not None else ori_loss

        if self.use_lipschitz:
            lipschitz_loss = self.model.decoder.lipschitz_loss()
            self.log(f"lipschitz_loss", lipschitz_loss.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            loss = loss + self.lipschitz_coeff * lipschitz_loss
        
        # Log losses
        self.log(f"val_loss", ori_loss.item(), on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if aux_loss is not None:
            self.log(f"val_aux_loss", aux_loss.item(), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        if batch_idx==0 and self.global_rank==0:
            idx = random.randrange(field.shape[0])
            self.log_image("val_visualization", field[idx], recon[idx])
        
        return loss

    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_steps,
            eta_min=0
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        } 
        
    def log_image(self, key: str, x: torch.Tensor, out: torch.Tensor):
        # x and out are already batch-indexed: shape (T, H, W, C)
        # Select timestep 1 to visualize
        x_vis = x.detach().float().cpu().numpy()[1]  # (H, W, C)
        out_vis = out.detach().float().cpu().numpy()[1]  # (H, W, C)
        diff_vis = np.abs(x_vis - out_vis)
        c = x_vis.shape[-1]

        fig, axs = plt.subplots(c, 3, figsize=(12, 3*c))
        
        if c == 1:
            axs = axs.reshape(1, -1)
        
        images = [x_vis, out_vis, diff_vis]
        titles = ["input", "recon", "abs_err"]
        
        for i in range(c):
            for j, (im, title) in enumerate(zip(images, titles)):
                ax = axs[i, j]
                
                if x_vis.ndim == 3:  # (H, W, C) - spatial data with channels
                    img_data = im[..., i]  # Select channel i -> (H, W)
                    img = ax.imshow(img_data, cmap="jet", aspect='auto')
                    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=8)
                elif x_vis.ndim == 2:  # (N, C) - 1D spatial data
                    ax.plot(im[:, i])
                    ax.set_xlabel("Coords")
                    ax.set_ylabel("Value")
                
                ax.set_title(f"{title} (ch {i})", fontsize=8)
        
        plt.tight_layout()

        self.logger.experiment.log({
                key: wandb.Image(fig, caption=f"epoch {self.current_epoch}"),
                "global_step": self.global_step
            })
        plt.close(fig)