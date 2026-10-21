"""PushT lowdim Priming workspace with isolated Stage-1/Stage-2 initialization.

Fresh Stage-2 post-training and opt-in Stage-1 MSE continuation may import
weights from the SAME trusted Stage-1 checkpoint while starting with a fresh
optimizer, EMA schedule, and zero global step. Resume still restores the
workspace's own latest checkpoint through the inherited run().
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
        stage = cfg.training.stage
        if stage not in ("pretrain", "posttrain"):
            raise ValueError("unsupported stage: " + str(stage))
        if self.model.stage != stage:
            raise ValueError("training.stage and policy.stage disagree")

        # If resume=true and our own latest checkpoint exists, let the
        # inherited runner load the COMPLETE training state. Otherwise, only
        # initialize policy weights from the requested external checkpoint.
        latest = self.get_checkpoint_path()
        resuming = bool(cfg.training.resume) and latest.is_file()
        if not resuming:
            source = cfg.training.init_checkpoint
            if stage == "posttrain" and not source:
                raise ValueError("posttrain requires training.init_checkpoint")
            if source:
                source_path = pathlib.Path(hydra.utils.to_absolute_path(source))
                if not source_path.is_file():
                    raise FileNotFoundError(
                        "missing initialization checkpoint: " + str(source_path))
                if cfg.training.init_weights not in ("ema", "online"):
                    raise ValueError("training.init_weights must be ema or online")
                # Only load user-trusted checkpoints: dill can execute code.
                payload = torch.load(
                    source_path, map_location="cpu", pickle_module=dill)
                saved = payload.get("state_dicts", {})
                source_key = (
                    "ema_model" if cfg.training.init_weights == "ema" else "model")
                if source_key not in saved:
                    raise KeyError("checkpoint lacks required weights: " + source_key)
                weights = saved[source_key]
                self.model.load_state_dict(weights, strict=True)
                if self.ema_model is not None:
                    self.ema_model.load_state_dict(weights, strict=True)
                # Intentionally do NOT load optimizer/global_step/epoch from
                # source: this is fresh fine-tuning, not checkpoint resume.
                print(f"Initialized {stage} from {source_path} [{source_key}]")

        # Inherited run still initializes dataset normalization, fresh
        # optimizer schedule, EMA updates, PushT rollout and checkpointing.
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
