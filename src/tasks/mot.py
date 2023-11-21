import subprocess
from pathlib import Path
from typing import Annotated

import luigi
from omegaconf import DictConfig


class MultipleObjectTracking(luigi.Task):
    scenario_id: Annotated[str, luigi.Parameter()] = luigi.Parameter()
    cfg: Annotated[DictConfig, luigi.Parameter()] = luigi.Parameter()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        Path(self.cfg.prediction_dir).mkdir(exist_ok=True)

    def output(self):
        return luigi.LocalTarget(f"{self.cfg.prediction_dir}/{self.scenario_id}.json")

    def run(self):
        cfg = self.cfg
        subprocess.run(
            [
                cfg.python,
                f"{cfg.project_root}/src/mot_strong_sort.py",
                f"{cfg.video_dir}/{self.scenario_id}/fp_video.mp4",
                "--detic-dump",
                f"{cfg.detic_dump_dir}/{self.scenario_id}.npy",
                "--output-json",
                self.output().path,
            ],
            check=True,
        )
