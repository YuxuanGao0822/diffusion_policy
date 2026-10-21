import pathlib
import hydra
from omegaconf import OmegaConf
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace

class TrainPrimingImageWorkspace(TrainDiffusionUnetImageWorkspace):
    def run(self):
        if self.cfg.policy.stage != "pretrain":
            raise ValueError("Image baseline only supports Stage 1")
        return super().run()

@hydra.main(version_base=None,
            config_path=str(pathlib.Path(__file__).parent.parent/"config"),
            config_name="train_priming_image_workspace")
def main(cfg: OmegaConf):
    TrainPrimingImageWorkspace(cfg).run()

if __name__ == "__main__":
    main()
