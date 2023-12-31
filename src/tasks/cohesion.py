import os
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import luigi
from omegaconf import DictConfig
from rhoknp import Document, Sentence

from tasks.util import FileBasedResourceManagerMixin
from utils.util import DatasetInfo


class CohesionAnalysis(luigi.Task, FileBasedResourceManagerMixin[int]):
    scenario_id: Annotated[str, luigi.Parameter()] = luigi.Parameter()
    cfg: Annotated[DictConfig, luigi.Parameter()] = luigi.Parameter()
    dataset_dir: Annotated[Path, luigi.PathParameter()] = luigi.PathParameter()

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
        cfg = self.cfg
        input_knp_file = Path(cfg.gold_knp_dir) / f"{self.scenario_id}.knp"

        if (gpu_id := self.acquire_resource()) is None:
            raise RuntimeError("No available GPU.")
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["DATA_DIR"] = f"{cfg.project_root}/data"
            prediction = self._run_cohesion(document=Document.from_knp(input_knp_file.read_text()), env=env)
            with self.output().open(mode="w") as f:
                f.write(prediction.to_knp())
        finally:
            self.release_resource(gpu_id)

    def _run_cohesion(self, document: Document, env: dict[str, str]) -> Document:
        dataset_info = DatasetInfo.from_json(self.dataset_dir.joinpath("info.json").read_text())
        sid2sentence: dict[str, Sentence] = {sentence.sid: sentence for sentence in document.sentences}
        cfg = self.cfg
        output_sentences: list[Sentence] = []
        for idx, utterance in enumerate(dataset_info.utterances):
            preceding_utterances = dataset_info.utterances[: idx + 1]
            preceding_document = Document.from_sentences(
                [sid2sentence[sid] for utterance in preceding_utterances for sid in utterance.sids]
            )

            with tempfile.TemporaryDirectory() as out_dir:
                doc_path = Path(out_dir).joinpath("caption.knp")
                doc_path.write_text(preceding_document.to_knp())
                subprocess.run(
                    [
                        cfg.python,
                        f"{cfg.project_root}/src/predict.py",
                        f"checkpoint={cfg.checkpoint}",
                        f"input_path={doc_path}",
                        f"export_dir={out_dir}",
                        "num_workers=0",
                        "devices=1",
                    ],
                    check=True,
                    env=env,
                )
                output_document = Document.from_knp(Path(out_dir).joinpath(f"{document.doc_id}.knp").read_text())

            output_sid2sentence: dict[str, Sentence] = {sent.sid: sent for sent in output_document.sentences}
            output_sentences += [output_sid2sentence[sid] for sid in utterance.sids]
        return Document.from_sentences(output_sentences)
