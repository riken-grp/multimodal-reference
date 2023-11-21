import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import luigi
from omegaconf import DictConfig


class CohesionAnalysis(luigi.Task):
    scenario_id: Annotated[str, luigi.Parameter()] = luigi.Parameter()
    cfg: Annotated[DictConfig, luigi.Parameter()] = luigi.Parameter()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        Path(self.cfg.prediction_dir).mkdir(exist_ok=True)

    def output(self):
        return luigi.LocalTarget(f"{self.cfg.prediction_dir}/{self.scenario_id}.knp")

    def run(self):
        cfg = self.cfg
        input_knp_file = Path(cfg.gold_knp_dir) / f"{self.scenario_id}.knp"
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
            )
            output_knp_file = next(Path(out_dir).glob("*.knp"))
            shutil.copy(output_knp_file, self.output().path)
