import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import luigi
from omegaconf import DictConfig

from utils.mot import DetectionLabels


class MultipleObjectTracking(luigi.Task):
    scenario_id: Annotated[str, luigi.Parameter()] = luigi.Parameter()
    cfg: Annotated[DictConfig, luigi.DictParameter()] = luigi.DictParameter()

    def requires(self):
        pass

    def output(self):
        return luigi.LocalTarget(f"{self.cfg.prediction_dir}/{self.scenario_id}.json")

    def run(self):
        prediction = run_mot(self.cfg, scenario_id=self.scenario_id)
        with self.output().open(mode="w") as f:
            f.write(prediction.to_json(ensure_ascii=False, indent=2))


def run_mot(cfg: DictConfig, scenario_id: str) -> DetectionLabels:
    with tempfile.TemporaryDirectory() as out_dir:
        subprocess.run(
            [
                cfg.python,
                f"{cfg.project_root}/src/mot_strong_sort.py",
                f"{cfg.video_dir}/{scenario_id}/fp_video.mp4",
                "--detic-dump",
                f"{cfg.detic_dump_dir}/{scenario_id}.npy",
                "--output-json",
                f"{out_dir}/mot.json",
            ],
            check=True,
        )
        return DetectionLabels.from_json(Path(out_dir).joinpath("mot.json").read_text())
