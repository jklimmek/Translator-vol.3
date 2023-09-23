import torch.nn as nn
import torch.optim as optim
from numpy import array
from pytorch_lightning import LightningModule
from xformers.factory.model_factory import xFormer, xFormerConfig
from .schedulers import *


class Transformer(LightningModule):
    def __init__(self, architecture, optimizer, config):
        super().__init__()

        xformer_config = xFormerConfig(
            stack_configs = [architecture["encoder"], architecture["decoder"]],
            tie_embedding_weights = architecture["tie_embedding_weights"],
            weight_init = architecture["weight_init"]
        )

        self.model = nn.ModuleDict(
            {
                "xformer": xFormer.from_config(xformer_config),
                "decoders_final_norm": nn.LayerNorm(architecture["dim"]),
                "decoders_out": nn.Linear(
                    architecture["dim"], 
                    architecture["vocab_size"], 
                    bias=architecture["out_bias"]
                )
            }
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
        self.architecture = architecture
        self.optimizer_config = optimizer
        self.config = config

        if self.config["log_diagnostics_every_n_steps"] > 0:
            self.encoder_weights = {}
            self.decoder_weights = {}

    
    def forward(self, src, tgt):
        x = self.model["xformer"](src, tgt)
        x = self.model["decoders_final_norm"](x)
        x = self.model["decoders_out"](x)
        return x
    

    def _common_step(self, batch):
        src, tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        logits = self(src, tgt_input)
        loss = self.criterion(logits.permute(0, 2, 1), tgt_output)
        return loss


    def training_step(self, batch, _):
        loss = self._common_step(batch)
        self._log_weights_and_updates()
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss


    def validation_step(self, batch, _):
        loss = self._common_step(batch)
        self.log("dev_loss", loss, on_step=True, on_epoch=True)
        return loss
    

    def configure_optimizers(self):
        decay = []
        no_decay = []

        no_decay_layers = self.optimizer_config.pop("no_decay_layers")
        for name, param in self.model.named_parameters():
            if any(nd in name for nd in no_decay_layers):
                no_decay.append(param)
            else:
                decay.append(param)

        OPTIMIZERS = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "radam": optim.RAdam,
        }

        SCHEDULERS = {
            "cosine": Cosine,
            "inv_sqrt_decay": InvSqrtDecay,
        }

        optim_groups = [
            {"params": decay, "weight_decay": self.optimizer_config.pop("weight_decay")},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer = OPTIMIZERS[self.optimizer_config.pop("optimizer")](
            optim_groups,
            lr = self.optimizer_config.pop("lr"),
            betas = self.optimizer_config.pop("betas"),
            eps = self.optimizer_config.pop("eps")
        )

        scheduler = {
            "scheduler": SCHEDULERS[self.optimizer_config.pop("scheduler")](
                optimizer, 
                **self.optimizer_config
            ),
            "interval": "step",
            "frequency": 1,
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
    

    def _log_weights_and_updates(self):
        """
        Logs weights and weight updates to TensorBoard.
        However, this substantially slows down training.
        """

        # Get interval and hheck if logging is enabled and if it is time to log.
        log_interval = self.config["log_diagnostics_every_n_steps"]
        if log_interval == 0 or (self.global_step + 1) % log_interval != 0:
            return

        # Get TensorBoard logger.
        tensorboard = self.logger.experiment

        # Initialize encoder and decoder weights.
        encoder_weights = {}
        decoder_weights = {}

        # Iterate over all model parameters.
        for name, param in self.model.named_parameters():

            # Get only encoder and decoder weights.
            if "xformer.encoders" in name or "xformer.decoders" in name and "norm" not in name and "bias" not in name:

                # Set prefix and layer number.
                prefix = "encoder" if "encoders" in name else "decoder"
                layer_num = int(name.split(".")[2])

                # Get weights and flatten them.
                weights = param.detach().flatten().cpu().tolist()

                # Append weights to encoder or decoder weights.
                if prefix == "encoder":
                    encoder_weights.setdefault(layer_num, []).extend(weights)
                elif prefix == "decoder":
                    decoder_weights.setdefault(layer_num, []).extend(weights)

        # Iterate over encoder and decoder weights.
        for prefix, weights_dict, old_weights_dict in [("encoder", encoder_weights, self.encoder_weights), 
                                                    ("decoder", decoder_weights, self.decoder_weights)]:
            
            # Iterate over layers.
            for layer_num, weights in weights_dict.items():

                # Log weights.
                if weights is not None and layer_num is not None:
                    tensorboard.add_histogram(f"{prefix}/layer{layer_num}", array(weights), self.global_step)

                # Log weight updates.
                old_weights = old_weights_dict.get(layer_num)
                if old_weights is not None:
                    update_magnitude = abs(array(weights) - array(old_weights)).mean()
                    tensorboard.add_scalar(f"{prefix}/update_layer{layer_num}", update_magnitude, self.global_step)

        # Update encoder and decoder weights for next logging.
        self.encoder_weights = encoder_weights.copy()
        self.decoder_weights = decoder_weights.copy()
