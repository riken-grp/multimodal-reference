import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import luigi
from omegaconf import DictConfig

from tasks.util import FileBasedResourceManagerMixin


class GLIPObjectDetection(luigi.Task, FileBasedResourceManagerMixin[int]):
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
                    f"{cfg.project_root}/tools/run_glip_video.py",
                    f"--model={cfg.checkpoint}",
                    f"--config-file={cfg.config}",
                    f"--video-input={input_video_file.resolve()}",
                    f"--confidence-threshold={cfg.confidence_threshold}",
                    f"--output={self.output().path}",
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
