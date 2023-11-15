import subprocess
import tempfile
from pathlib import Path

import luigi
from omegaconf import DictConfig
from rhoknp import Document


class CohesionAnalysis(luigi.Task):
    scenario_id = luigi.Parameter()
    cfg = luigi.Parameter()

    # def __init__(self, cfg: DictConfig) -> None:
    #     super().__init__()
    #     self.cfg = cfg

    def requires(self):
        pass

    def output(self):
        return luigi.LocalTarget(f"{self.cfg.prediction_dir}/{self.scenario_id}.knp")

    def run(self):
        document = run_cohesion(self.cfg, Path(self.cfg.gold_knp_dir) / f"{self.scenario_id}.knp")
        with self.output().open(mode="w") as f:
            f.write(document.to_knp())


def run_cohesion(cfg: DictConfig, input_knp_file: Path) -> Document:
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
        return Document.from_knp(next(Path(out_dir).glob("*.knp")).read_text())
