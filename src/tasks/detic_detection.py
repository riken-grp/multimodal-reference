import os
import socket
import subprocess
from pathlib import Path
from typing import Annotated

import luigi
from omegaconf import DictConfig

from tasks.util import FileBasedResourceManagerMixin


class DeticObjectDetection(luigi.Task, FileBasedResourceManagerMixin[int]):
    scenario_id: Annotated[str, luigi.Parameter()] = luigi.Parameter()
    cfg: Annotated[DictConfig, luigi.Parameter()] = luigi.Parameter()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        available_gpus = [int(gpu_id) for gpu_id in os.environ.get("AVAILABLE_GPUS", "0").split(",")]
        super(luigi.Task, self).__init__(
            available_gpus, Path("shared_state.json"), state_prefix=f"{socket.gethostname()}_gpu"
        )
        Path(self.cfg.prediction_dir).mkdir(exist_ok=True)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(Path(self.cfg.prediction_dir).resolve() / f"{self.scenario_id}.npy")

    def run(self):
        cfg = self.cfg
        input_video_file = Path(cfg.recording_dir) / self.scenario_id / "fp_video.mp4"

        gpu_id = self.acquire_resource()
        if gpu_id is None:
            raise RuntimeError("No available GPU.")
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            subprocess.run(
                [
                    cfg.python,
                    f"{cfg.project_root}/export.py",
                    f"--config-file={cfg.config}",
                    f"--video-input={input_video_file.resolve()}",
                    "--vocabulary=lvis",
                    f"--confidence-threshold={cfg.confidence_threshold}",
                    f"--output={self.output()}",
                    "--opts",
                    "MODEL.WEIGHTS",
                    cfg.model,
                ],
                cwd=cfg.project_root,
                env=env,
                check=True,
            )
        finally:
            self.release_resource(gpu_id)
