import subprocess
from pathlib import Path
from typing import Annotated

import luigi
from omegaconf import DictConfig

from tasks.detic_detection import DeticObjectDetection


class MultipleObjectTracking(luigi.Task):
    scenario_id: Annotated[str, luigi.Parameter()] = luigi.Parameter()
    cfg: Annotated[DictConfig, luigi.Parameter()] = luigi.Parameter()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        Path(self.cfg.prediction_dir).mkdir(exist_ok=True, parents=True)

    def requires(self) -> luigi.Task:
        return DeticObjectDetection(scenario_id=self.scenario_id, cfg=self.cfg.detic_detection)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(f"{self.cfg.prediction_dir}/{self.scenario_id}.json")

    def complete(self) -> bool:
        if not Path(self.output().path).exists():
            return False

        self_mtime = Path(self.output().path).stat().st_mtime
        task = self.requires()
        if not task.complete():
            return False
        output = task.output()
        assert isinstance(output, luigi.LocalTarget), f"output is not LocalTarget: {output}"
        if Path(output.path).stat().st_mtime > self_mtime:
            return False

        return True

    def run(self) -> None:
        cfg = self.cfg
        input_video_file = Path(cfg.recording_dir) / self.scenario_id / "fp_video.mp4"
        subprocess.run(
            [
                cfg.python,
                f"{cfg.project_root}/src/mot_strong_sort.py",
                input_video_file,
                f"--detic-dump={self.input().path}",
                f"--output-json={self.output().path}",
            ],
            check=True,
        )
