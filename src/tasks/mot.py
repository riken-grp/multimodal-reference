import subprocess
from pathlib import Path
from typing import Annotated

import luigi
from omegaconf import DictConfig
from rhoknp import Document

from tasks import DeticPhraseGrounding


class MultipleObjectTracking(luigi.Task):
    scenario_id: Annotated[str, luigi.Parameter()] = luigi.Parameter()
    cfg: Annotated[DictConfig, luigi.Parameter()] = luigi.Parameter()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dataset_dir = Path(self.cfg.dataset_dir) / self.scenario_id
        self.gold_document = Document.from_knp(
            Path(self.cfg.gold_knp_dir).joinpath(f"{self.scenario_id}.knp").read_text()
        )
        Path(self.cfg.prediction_dir).mkdir(exist_ok=True)

    def requires(self) -> luigi.Task:
        return DeticPhraseGrounding(
            scenario_id=self.scenario_id,
            cfg=self.cfg.detic,
            document=self.gold_document,
            dataset_dir=self.dataset_dir,
        )

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(f"{self.cfg.prediction_dir}/{self.scenario_id}.json")

    def run(self) -> None:
        cfg = self.cfg.mot
        subprocess.run(
            [
                cfg.python,
                f"{cfg.project_root}/src/mot_strong_sort.py",
                f"{cfg.recording_dir}/{self.scenario_id}/fp_video.mp4",
                "--detic-dump",
                f"{cfg.detic_dump_dir}/{self.scenario_id}.npy",
                "--output-json",
                self.output().path,
            ],
            check=True,
        )
