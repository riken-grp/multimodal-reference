import base64
import math
import pickle
import sys
import textwrap
from pathlib import Path
from typing import Annotated

import luigi
import numpy as np
import requests
from omegaconf import DictConfig
from openai import OpenAI
from rhoknp import Document, Sentence

from utils.prediction import BoundingBox as BoundingBoxPrediction
from utils.prediction import (
    PhraseGroundingPrediction,
    PhrasePrediction,
    RelationPrediction,
    UtterancePrediction,
)
from utils.util import DatasetInfo, Rectangle


class SoMPhraseGrounding(luigi.Task):
    scenario_id: Annotated[str, luigi.Parameter()] = luigi.Parameter()
    cfg: Annotated[DictConfig, luigi.Parameter()] = luigi.Parameter()
    document_path: Annotated[Path, luigi.Parameter()] = luigi.PathParameter()
    dataset_dir: Annotated[Path, luigi.PathParameter()] = luigi.PathParameter()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        Path(self.cfg.prediction_dir).mkdir(exist_ok=True, parents=True)

    # def requires(self) -> luigi.Task:
    #     return hydra.utils.instantiate(self.cfg.detection, scenario_id=self.scenario_id)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(f"{self.cfg.prediction_dir}/{self.scenario_id}.json")

    def run(self) -> None:
        with self.input().open(mode="r") as f:
            detection_dump: list[np.ndarray] = pickle.load(f)

        utterance_predictions: list[UtterancePrediction] = []
        document = Document.from_knp(self.document_path.read_text())
        dataset_info = DatasetInfo.from_json(self.dataset_dir.joinpath("info.json").read_text())
        sid2sentence: dict[str, Sentence] = {sentence.sid: sentence for sentence in document.sentences}
        for idx, utterance in enumerate(dataset_info.utterances):
            start_index = math.ceil(utterance.start / 1000)
            if idx + 1 < len(dataset_info.utterances):
                next_utterance = dataset_info.utterances[idx + 1]
                end_index = math.ceil(next_utterance.start / 1000)
            else:
                end_index = len(dataset_info.images)
            corresponding_images = dataset_info.images[start_index:end_index]
            caption = Document.from_sentences([sid2sentence[sid] for sid in utterance.sids])
            phrase_predictions: list[PhrasePrediction] = [
                PhrasePrediction(
                    sid=base_phrase.sentence.sid,
                    index=base_phrase.global_index,
                    text=base_phrase.text,
                    relations=[],
                )
                for base_phrase in caption.base_phrases
            ]
            for phrase_prediction in phrase_predictions:
                for image in corresponding_images:
                    image_idx = int(image.id) - 1
                    raw_bbs: np.ndarray = detection_dump[image_idx * 30]  # (bb, 6)
                    for raw_bb in raw_bbs.tolist():
                        phrase_prediction.relations.append(
                            RelationPrediction(
                                type="=",
                                image_id=image.id,
                                bounding_box=BoundingBoxPrediction(
                                    image_id=image.id,
                                    rect=Rectangle(
                                        x1=raw_bb[0] / 2, y1=raw_bb[1] / 2, x2=raw_bb[2] / 2, y2=raw_bb[3] / 2
                                    ),
                                    confidence=raw_bb[4],
                                ),
                            )
                        )

            utterance_predictions.append(
                UtterancePrediction(text=caption.text, sids=utterance.sids, phrases=phrase_predictions)
            )

        prediction = PhraseGroundingPrediction(
            scenario_id=self.scenario_id,
            images=dataset_info.images,
            utterances=utterance_predictions,
        )

        with self.output().open(mode="w") as f:
            f.write(prediction.to_json(ensure_ascii=False, indent=2))


def _get_prompt(context_utterances: list, target_utterance) -> str:
    prompt = textwrap.dedent(
        """\
        I have labeled a bright numeric ID at the center for each visual object in the image.
        Given the image and an related utterance and its dialogue context,
        find the corresponding regions for 「ここ」, 「棚」, 「下」 if any.
        The response must be a JSON format like the following:
        {'ここ': [1], '棚': [4], '下': null}

        ## Dialogue Context
        """
    )
    for utterance in context_utterances:
        prompt += f"{utterance.speaker}: {utterance.text}\n"
    prompt += "## Target Utterance\n"
    prompt += f"{target_utterance.speaker}: {target_utterance.text}\n"
    return prompt


def main() -> None:
    dataset_dir = Path("data/dataset/20220302-56133195-2/info.json")
    dataset_info = DatasetInfo.from_json(dataset_dir.read_text())
    image_path = dataset_dir / "images" / "005.png"
    with image_path.open(mode="rb") as f:
        base64_str = base64.b64encode(f.read()).decode("utf-8")

    try:
        response = requests.post("http://localhost:6093/mark", json={"base64_str": base64_str})
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as e:
        print(e, file=sys.stderr)
    except Exception as e:
        print(e, file=sys.stderr)

    context_utterances = dataset_info.utterances[:10]
    utterance = dataset_info.utterances[10]
    prompt = _get_prompt(context_utterances, utterance)

    print(result)
    marked_image_b64_str = result["base64_str"]
    img_type = "png"
    client = OpenAI()
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img_type};base64,{marked_image_b64_str}"},
                    },
                ],
            }
        ],
        model="gpt-4o-2024-08-06",
    )

    print(response)


if __name__ == "__main__":
    main()
