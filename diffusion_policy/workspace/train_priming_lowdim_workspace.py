"""Reuse the original PushT lowdim training/rollout workspace unchanged.

For post-training, import model weights from a Stage-1 checkpoint but
start a *fresh optimizer and training schedule*. Checkpoints saved by this
workspace retain the normal Diffusion Policy BaseWorkspace payload format.
"""
import pathlib

import dill
import hydra
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import (
    TrainDiffusionUnetLowdimWorkspace,
)


class TrainPrimingLowdimWorkspace(TrainDiffusionUnetLowdimWorkspace):
    def run(self):
        cfg = self.cfg
        if cfg.training.stage == "posttrain":
            if self.model.stage != "posttrain":
                raise ValueError("training.stage and policy.stage disagree")
            latest = self.get_checkpoint_path()
            resuming = bool(cfg.training.resume) and latest.is_file()
            if not resuming:
                source = cfg.training.init_checkpoint
                if not source:
                    raise ValueError("posttrain requires training.init_checkpoint")
                source_path = pathlib.Path(hydra.utils.to_absolute_path(source))
                if not source_path.is_file():
                    raise FileNotFoundError("missing pretraining checkpoint: " + str(source_path))
                # Only load a checkpoint that you trust: dill can deserialize code.
                payload = torch.load(source_path, map_location="cpu", pickle_module=dill)
                saved = payload.get("state_dicts", {})
                if cfg.training.init_weights not in ("ema", "online"):
                    raise ValueError("training.init_weights must be ema or online")
                source_key = "ema_model" if cfg.training.init_weights == "ema" else "model"
                if source_key not in saved:
                    raise KeyError("checkpoint lacks required weights: " + source_key)
                weights = saved[source_key]
                # Both online and EMA weights start identically at posttrain step 0.
                self.model.load_state_dict(weights, strict=True)
                if self.ema_model is not None:
                    self.ema_model.load_state_dict(weights, strict=True)
                print(f"Initialized post-training from {source_path} [{source_key}]")
        elif cfg.training.stage != "pretrain":
            raise ValueError("unsupported stage: " + str(cfg.training.stage))
        elif self.model.stage != "pretrain":
            raise ValueError("training.stage and policy.stage disagree")
        # The inherited run provides data normalization, optimizer, EMA,
        # validation, PushT rollouts, metrics and checkpointing.
        return super().run()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent / "config"),
    config_name="train_priming_lowdim_workspace",
)
def main(cfg: OmegaConf):
    workspace = TrainPrimingLowdimWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
