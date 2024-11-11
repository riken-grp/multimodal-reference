import base64
import io
import json
import math
import sys
import textwrap
from pathlib import Path
from typing import Annotated, Any, Optional

import luigi
import PIL.Image
import requests
from omegaconf import DictConfig
from openai import OpenAI
from PIL.Image import Image
from pydantic import BaseModel
from rhoknp import BasePhrase, Document, Sentence
from tenacity import retry, stop_after_attempt, wait_fixed

from utils.prediction import (
    BoundingBox,
    PhraseGroundingPrediction,
    PhrasePrediction,
    RelationPrediction,
    UtterancePrediction,
)
from utils.util import DatasetInfo, Rectangle, UtteranceInfo, get_core_expression

SOM_API_URL = "http://moss110:6093/mark"


class Phrase(BaseModel):
    index: int
    text: str
    referred_object_ids: list[int]


class PhraseGroundingResult(BaseModel):
    phrases: list[Phrase]


class SoMPhraseGrounding(luigi.Task):
    scenario_id: Annotated[str, luigi.Parameter()] = luigi.Parameter()
    cfg: Annotated[DictConfig, luigi.Parameter()] = luigi.Parameter()
    document_path: Annotated[Path, luigi.Parameter()] = luigi.PathParameter()
    dataset_dir: Annotated[Path, luigi.PathParameter()] = luigi.PathParameter()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        Path(self.cfg.prediction_dir).mkdir(exist_ok=True, parents=True)
        self.mark_dir = Path(self.cfg.prediction_dir) / self.scenario_id
        self.client = OpenAI()

    def requires(self) -> luigi.Task:
        return SoMMarking(
            scenario_id=self.scenario_id, cfg=self.cfg, document_path=self.document_path, dataset_dir=self.dataset_dir
        )

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(f"{self.cfg.prediction_dir}/{self.scenario_id}.json")

    def run(self) -> None:
        image_id_to_masks: dict[str, list[dict[str, Any]]] = json.loads(self.input().open().read())
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
            corresponding_image_ids: list[str] = [
                image_info.id for image_info in dataset_info.images[start_index:end_index]
            ]
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
            prompt = _get_prompt(dataset_info.utterances[:idx], utterance, caption.base_phrases)
            print(prompt)
            for image_id in corresponding_image_ids:
                label_to_mask = {mask["label"]: mask for mask in image_id_to_masks[image_id]}
                image_file = self.mark_dir / "images" / f"{image_id}.png"
                result = self._call_openai(image_file, prompt)
                if result is None:
                    continue
                assert isinstance(result, PhraseGroundingResult)
                print(image_file)
                print(result)
                for grounded_phrase in result.phrases:
                    if grounded_phrase.index < 0 or grounded_phrase.index >= len(phrase_predictions):
                        print(f"Warning: Invalid index {grounded_phrase.index}", file=sys.stderr)
                        continue
                    phrase_prediction = phrase_predictions[grounded_phrase.index]
                    if grounded_phrase.text not in phrase_prediction.text:
                        print(f"Warning: {grounded_phrase.text} is not in {phrase_prediction.text}", file=sys.stderr)
                    for object_id in grounded_phrase.referred_object_ids:
                        label = str(object_id)
                        if label not in label_to_mask:
                            continue
                        mask = label_to_mask[label]
                        phrase_prediction.relations.append(
                            RelationPrediction(
                                type="=",
                                image_id=image_id,
                                bounding_box=BoundingBox(
                                    image_id=image_id,
                                    rect=Rectangle.from_xywh(*mask["bbox"]),
                                    confidence=1.0,
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

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(60 * 10))
    def _call_openai(self, image_file: Path, prompt: str) -> Optional[PhraseGroundingResult]:
        body = {
            "model": self.cfg.openai_model,
            "messages": _gen_messages(image_file, prompt),
            "response_format": PhraseGroundingResult,
        }
        completion = self.client.beta.chat.completions.parse(**body)
        message = completion.choices[0].message
        if message.refusal:
            print(message.refusal)
            return None
        return message.parsed


class SoMMarking(luigi.Task):
    scenario_id: Annotated[str, luigi.Parameter()] = luigi.Parameter()
    cfg: Annotated[DictConfig, luigi.Parameter()] = luigi.Parameter()
    document_path: Annotated[Path, luigi.Parameter()] = luigi.PathParameter()
    dataset_dir: Annotated[Path, luigi.PathParameter()] = luigi.PathParameter()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mark_dir = Path(self.cfg.prediction_dir) / self.scenario_id
        self.mark_dir.mkdir(exist_ok=True, parents=True)
        self.marked_image_dir = self.mark_dir / "images"
        self.marked_image_dir.mkdir(exist_ok=True, parents=True)

    def output(self) -> luigi.LocalTarget:
        return luigi.LocalTarget(f"{self.mark_dir}/masks.json")

    def run(self) -> None:
        masks: dict[str, list] = {}
        for image_path in sorted(self.dataset_dir.glob("images/*.png")):
            with image_path.open(mode="rb") as f:
                base64_str = base64.b64encode(f.read()).decode("utf-8")

            response = requests.post(
                SOM_API_URL,
                json={"base64_str": base64_str, "granularity": self.cfg.segmentation_granularity},
            )
            response.raise_for_status()
            result = response.json()

            image = base64_to_pil(result["base64_str"])
            image.save(f"{self.marked_image_dir}/{image_path.name}")

            masks[image_path.stem] = result["masks"]

        with self.output().open(mode="w") as f:
            f.write(json.dumps(masks, ensure_ascii=False, indent=2))


def base64_to_pil(base64_str: str) -> Image:
    if ";base64," in base64_str:
        base64_str = base64_str.split(";base64,")[1]

    img_data = base64.b64decode(base64_str)
    img_buffer = io.BytesIO(img_data)
    img = PIL.Image.open(img_buffer)
    return img


def pil_to_base64(image: Image, format: str = "PNG") -> str:
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _gen_messages(image_file: Path, user_prompt: str) -> list[dict[str, Any]]:
    system_prompt = _get_system_prompt()
    with image_file.open(mode="rb") as f:
        marked_image_b64_str = base64.b64encode(f.read()).decode("utf-8")
    media_type = "image/png"
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{marked_image_b64_str}"},
                },
            ],
        },
    ]
    return messages


def _get_system_prompt() -> str:
    return textwrap.dedent(
        """\
        You are a super-intelligent household robot that helps your master with daily mundane tasks.
        Given a dialogue context, target utterance, and a household scene image, identify specific objects or areas referenced by each phrase in the target utterance. This task is also known as phrase grounding.
        Each visual object in the image is labeled with a bright numeric ID at its center. Your task is to output the corresponding object ID for each target phrase where an object is referenced.

        Note the following key points:
        - Noisy Target Phrases: Some target phrases include verbs, adverbs, or interjections that do not reference any physical object. For such phrases, return an empty object ID.
        - Object Availability: Even if a target phrase uses a noun or otherwise implies a physical object, the referenced object may not appear in the image. In these cases, return an empty object ID.

        Assume that the scene could occur in various household settings (e.g., living room, dining area, kitchen), with a range of typical household items (e.g., cleaning tools, kitchen utensils, furniture). Use both visual and linguistic clues to determine references within the image.
        """
    )


def _get_prompt(
    context_utterances: list[UtteranceInfo], target_utterance: UtteranceInfo, base_phrases: list[BasePhrase]
) -> str:
    core_expressions = [get_core_expression(base_phrase) for base_phrase in base_phrases]
    prompt = textwrap.dedent(
        """\
        Analyze the following dialogue context and target utterance (「{}」) to determine the corresponding object IDs for each target phrase.
        Each visual object in the image is labeled with a bright numeric ID at its center.

        For each target phrase:
        - Identify and provide the numeric object ID(s) if the phrase references one or more objects visible in the image.
        - Leave the object ID field empty if the phrase does not reference a physical object or if the referenced object is not visible in the image.

        ## Dialogue Context
        """
    ).format(target_utterance.text)
    for utterance in context_utterances:
        prompt += f"{utterance.speaker}: {utterance.text}\n"
    prompt += "\n## Target Utterance\n"
    prompt += f"{target_utterance.speaker}: {target_utterance.text}\n"
    prompt += "\n## Target Phrases\n"
    for idx, core_expression in enumerate(core_expressions):
        prompt += f"{idx}. {core_expression[1]}\n"
    return prompt
