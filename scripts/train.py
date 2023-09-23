import argparse
import os
import warnings

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (Callback, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import TensorBoardLogger

from .dataloader import TranslatorDataLoader
from .dataset import TranslatorDataset
from .model import Transformer
from .utils import get_model_size, params_check, read_yaml, seed_all

# Filter out warnings.
warnings.filterwarnings("ignore", message="Checkpoint directory .* exists and is not empty.")
warnings.filterwarnings("ignore", message=".*num_workers.*")


# import tensorboard

class GradientNormLoggerCallback(Callback):
    def __init__(self, log_interval=5):
        super().__init__()
        self.log_interval = log_interval
        self.encoder_norms = []
        self.decoder_norms = []

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        if (trainer.global_step + 1) % self.log_interval == 0:
            self.encoder_norms = []
            self.decoder_norms = []

            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    if "encoders" in name:
                        self.encoder_norms.append(torch.norm(param.grad, p=2).item())

                    elif "decoders" in name:
                        self.decoder_norms.append(torch.norm(param.grad, p=2).item())

            if len(self.encoder_norms) != 0:
                enc_avg_norm = sum(self.encoder_norms) / len(self.encoder_norms)
                pl_module.log("norms/encoder_norm", enc_avg_norm, on_step=True, on_epoch=False)
            
            if len(self.decoder_norms) != 0:
                dec_avg_norm = sum(self.decoder_norms) / len(self.decoder_norms)
                pl_module.log("norms/decoder_norm", dec_avg_norm, on_step=True, on_epoch=False)


def run_training(architecture, optimizer, config, comment):
    seed_all(config["seed"])
    
    params_check(architecture)

    data_loader = TranslatorDataLoader(
        train_dataset = torch.load(config["train_dataset"]),
        dev_dataset = torch.load(config["dev_dataset"]),
        batch_size = config["batch_size"],
        num_workers = config["num_workers"]
    )

    model = Transformer(architecture, optimizer, config)
    model_size = get_model_size(model)
    training_name = f"{model_size}_{comment}" if comment is not None else model_size

    checkpoint_dir = os.path.join(config["save_dir"], model_size)
    logs_dir = os.path.join(config["logs_dir"], model_size)

    logger = TensorBoardLogger(
        save_dir = logs_dir, 
        name = training_name,
        default_hp_metric = False,
        version = ""
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath = checkpoint_dir,
        filename = "{epoch:02d}-{train_loss:.4f}-{dev_loss:.4f}",
        save_top_k = config["save_top_k"],
    )

    lr_monitor_callback = LearningRateMonitor(logging_interval="step")

    gradient_norm_logger_callback = GradientNormLoggerCallback(log_interval=config["log_every_n_steps"])

    trainer = Trainer(
        max_epochs = config["total_epochs"],
        accumulate_grad_batches = config["accumulation_steps"],
        log_every_n_steps = config["log_every_n_steps"],
        gradient_clip_algorithm = config["gradient_clip_algorithm"],
        gradient_clip_val = config["gradient_clip_val"],
        precision = config["precision"],
        logger = logger,
        callbacks = [
            lr_monitor_callback, 
            checkpoint_callback, 
            gradient_norm_logger_callback
        ],
    )

    trainer.fit(model, data_loader, ckpt_path=config["checkpoint"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Transformer Translator."
    )
    parser.add_argument("--architecture", type=str, required=True, help="Path to architecture YAML file.")
    parser.add_argument("--optimizer", type=str, required=True, help="Path to optimizer YAML file.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file.")
    parser.add_argument("--comment", type=str, default=None, help="Additional comment for Tensorbord.")
    return parser.parse_args()


def main():
    args = parse_args()
    architecture = read_yaml(args.architecture)
    optimizer = read_yaml(args.optimizer)
    config = read_yaml(args.config)

    run_training(
        architecture = architecture,
        optimizer = optimizer,
        config = config,
        comment=args.comment,
    )
    

if __name__ == "__main__":
    main()