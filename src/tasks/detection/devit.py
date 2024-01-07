import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import luigi
from omegaconf import DictConfig

from tasks.util import FileBasedResourceManagerMixin


class DEViTObjectDetection(luigi.Task, FileBasedResourceManagerMixin[int]):
    scenario_id: Annotated[str, luigi.Parameter()] = luigi.Parameter()
    cfg: Annotated[DictConfig, luigi.Parameter()] = luigi.Parameter()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        available_gpus = [int(gpu_id) for gpu_id in os.environ.get("AVAILABLE_GPUS", "0").split(",")]
        super(luigi.Task, self).__init__(
            available_gpus, Path("shared_state.json"), state_prefix=f"{socket.gethostname()}_gpu"
        )
        Path(self.cfg.prediction_dir).mkdir(exist_ok=True, parents=True)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(
            Path(self.cfg.prediction_dir).resolve() / f"{self.scenario_id}.npy", format=luigi.format.Nop
        )

    def run(self):
        cfg = self.cfg
        input_video_file = Path(cfg.recording_dir) / self.scenario_id / "fp_video.mp4"

        if (gpu_id := self.acquire_resource()) is None:
            raise RuntimeError("No available GPU.")
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            subprocess.run(
                [
                    cfg.python,
                    f"{cfg.project_root}/demo/video.py",
                    f"--config-file={cfg.config}",
                    f"--rpn-config-file={cfg.rpn_config}",
                    f"--model-path={cfg.model}",
                    f"--video-file={input_video_file.resolve()}",
                    f"--output-file={self.output().path}",
                    "--category-space-file=None",
                    "--device=cuda",
                    f"--topk={cfg.class_topk}",
                    f"--threshold={cfg.confidence_threshold}",
                ],
                cwd=cfg.project_root,
                env=env,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(e.stderr, file=sys.stderr)
            raise e
        finally:
            self.release_resource(gpu_id)
