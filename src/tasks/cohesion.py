import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import luigi
from omegaconf import DictConfig

from tasks.util import FileBasedResourceManagerMixin


class CohesionAnalysis(luigi.Task, FileBasedResourceManagerMixin[int]):
    scenario_id: Annotated[str, luigi.Parameter()] = luigi.Parameter()
    cfg: Annotated[DictConfig, luigi.Parameter()] = luigi.Parameter()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        available_gpus = [int(gpu_id) for gpu_id in os.environ.get("AVAILABLE_GPUS", "0").split(",")]
        super(luigi.Task, self).__init__(
            available_gpus, Path("shared_state.json"), state_prefix=f"{socket.gethostname()}_gpu"
        )
        Path(self.cfg.prediction_dir).mkdir(exist_ok=True)

    def output(self):
        return luigi.LocalTarget(f"{self.cfg.prediction_dir}/{self.scenario_id}.knp")

    def run(self):
        gpu_id = self.acquire_resource()
        if gpu_id is None:
            raise RuntimeError("No available GPU.")
        cfg = self.cfg
        input_knp_file = Path(cfg.gold_knp_dir) / f"{self.scenario_id}.knp"

        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            with tempfile.TemporaryDirectory() as out_dir:
                subprocess.run(
                    [
                        cfg.python,
                        f"{cfg.project_root}/src/predict.py",
                        f"checkpoint={cfg.checkpoint}",
                        f"input_path={input_knp_file}",
                        f"export_dir={out_dir}",
                        "num_workers=0",
                        "devices=1",
                    ],
                    check=True,
                    env=env,
                )
                output_knp_file = next(Path(out_dir).glob("*.knp"))
                shutil.copy(output_knp_file, self.output().path)
        finally:
            self.release_resource(gpu_id)
