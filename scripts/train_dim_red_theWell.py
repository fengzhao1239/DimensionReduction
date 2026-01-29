import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
torch.set_float32_matmul_precision("high")
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from src.fcnf.network import SIRENautoencoder, TransolverAutoencoder
from src.engine.trainer_fcnf import FAETrainer
from src.utils.data_TheWell import get_data, get_dataloader


def get_wandb_run_name(cfg: DictConfig) -> str:
    """Create a descriptive wandb run name for tokenizer training."""
    # Get dataset name from path
    dataset_name = cfg.data.dataset_name
    
    # Get model parameters
    basic_info = [f"Ld{cfg.model.latent_features}",
                  f"E{cfg.model.encoder_num_hidden_layers}x{cfg.model.encoder_hidden_features}",
                  f"D{cfg.model.decoder_num_hidden_layers}x{cfg.model.decoder_hidden_features}",]
                #   f"lip{cfg.model.use_lipschitz}"]
    
    quantizer_info = ["NoQuantizer"]
    if "quantizer_cfg" in cfg.model:
        if "levels" in cfg.model.quantizer_cfg:
            quantizer_info = ["FSQ",
                              f"{cfg.model.quantizer_cfg.levels}x{cfg.model.quantizer_cfg.num_codebooks}",]
        else:
            quantizer_info = ["VQ",
                              f"cbs{cfg.model.quantizer_cfg.codebook_size}",
                              f"h{cfg.model.quantizer_cfg.heads}x{cfg.model.quantizer_cfg.codebook_dim}",]
        
    model_params = basic_info + quantizer_info
    
    # Combine all parts
    return f"[DR-lowranklip]{dataset_name}_{'_'.join(model_params)}"


@hydra.main(version_base=None, config_path="../configs", config_name="dimension_reduction.yaml")
def train(cfg: DictConfig):
    # Set up logging
    L.seed_everything(cfg.training.seed)
    
    # Create logger with descriptive run name
    run_name = get_wandb_run_name(cfg)
    logger = WandbLogger(
        project=cfg.logging.project,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    dataset_name = cfg.data.dataset_name

    # load data
    train_dataset, val_dataset = get_data(base_path=cfg.data.data_dir,
                                          well_dataset_name=dataset_name,
                                          masking_strategy=cfg.data.masking_strategy,
                                          encoder_point_ratio=cfg.data.encoder_point_ratio,
                                          downsample_factor=cfg.data.downsample_factor)    #!tbd
    train_loader = get_dataloader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    val_loader = get_dataloader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    # Create model
    model = SIRENautoencoder(**cfg.model)

    # Create trainer
    trainer = FAETrainer(
        model=model,
        training_config=cfg.training
    )

    # Create callbacks
    # print(
    # "\n"
    # "================= Experiment Setup =================\n"
    # f"Logging directory : {cfg.logging.output_dir}\n"
    # f"Run name          : {run_name}\n"
    # "====================================================\n"
    # )
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(cfg.logging.output_dir, run_name),
            filename="{step}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            every_n_train_steps=cfg.training.save_every_n_train_steps,       # save every n steps
            save_on_train_epoch_end=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Create Lightning trainer
    pl_trainer = L.Trainer(
        max_steps=cfg.training.max_steps,
        # max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        strategy=cfg.training.strategy,
        precision=cfg.training.precision,
        # gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,    # validate every None epochs
        # val_check_interval=cfg.training.val_check_interval,
        callbacks=callbacks,
        logger=logger,
        # num_sanity_val_steps=0
    )

    # Train
    pl_trainer.fit(
        trainer,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        # ckpt_path="last"
    )

    # Save final model
    # if cfg.huggingface.push_to_hub:
    #     trainer.model.push_to_hub(
    #         repo_name=cfg.huggingface.repo_name,
    #         private=cfg.huggingface.private,
    #         commit_message=cfg.huggingface.commit_message,
    #         model_card=cfg.huggingface.model_card,
    #         model_card_template=cfg.huggingface.model_card_template,
    #     )

if __name__ == "__main__":
    train() 