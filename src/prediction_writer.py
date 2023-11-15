import copy
import itertools
import math
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Union

import hydra
from omegaconf import DictConfig
from rhoknp import BasePhrase, Document, Sentence
from rhoknp.cohesion import EndophoraArgument, RelMode, RelTagList
from rhoknp.cohesion.coreference import EntityManager

from utils.annotation import BoundingBox as BoundingBoxAnnotation
from utils.annotation import ImageAnnotation, ImageTextAnnotation
from utils.glip import GLIPPrediction
from utils.mdetr import MDETRPrediction
from utils.mot import DetectionLabels
from utils.prediction import BoundingBox as BoundingBoxPrediction
from utils.prediction import (
    PhraseGroundingPrediction,
    PhrasePrediction,
    RelationPrediction,
    UtterancePrediction,
)
from utils.util import DatasetInfo, box_iou


@hydra.main(version_base=None, config_path="../configs")
def main(cfg: DictConfig) -> None:
    dataset_dir = Path(cfg.dataset_dir)
    gold_knp_file = Path(cfg.gold_knp_file)
    gold_annotation_file = Path(cfg.gold_annotation_file)
    prediction_dir = Path(cfg.prediction_dir)
    prediction_dir.mkdir(exist_ok=True)

    dataset_info = DatasetInfo.from_json(dataset_dir.joinpath("info.json").read_text())
    scenario_id = dataset_info.scenario_id
    gold_document = Document.from_knp(gold_knp_file.read_text())
    gold_annotation = ImageTextAnnotation.from_json(gold_annotation_file.read_text())

    pred_knp_file = prediction_dir / f"{scenario_id}.knp"
    assert pred_knp_file.exists()
    parsed_document = Document.from_knp(pred_knp_file.read_text())

    # perform phrase grounding with MDETR
    phrase_grounding_file = prediction_dir / "mdetr" / f"{scenario_id}.json"
    phrase_grounding_file.parent.mkdir(exist_ok=True)
    assert phrase_grounding_file.exists()
    phrase_grounding_prediction = PhraseGroundingPrediction.from_json(phrase_grounding_file.read_text())

    parsed_document = preprocess_document(parsed_document)
    mm_reference_prediction = relax_prediction_with_pas_bridging(phrase_grounding_prediction, parsed_document)

    if cfg.coref_relax_mode == "pred":
        relax_prediction_with_coreference(phrase_grounding_prediction, parsed_document)
    elif cfg.coref_relax_mode == "gold":
        relax_prediction_with_coreference(phrase_grounding_prediction, gold_document)

    if cfg.mot_relax_mode == "pred":
        mot_file = prediction_dir / "mot" / f"{scenario_id}.json"
        if not mot_file.exists():
            mot_result = run_mot(cfg.mot, scenario_id)
            mot_file.parent.mkdir(exist_ok=True)
            mot_file.write_text(mot_result.to_json(ensure_ascii=False, indent=2))
        else:
            mot_result = DetectionLabels.from_json(mot_file.read_text())
        relax_prediction_with_mot(phrase_grounding_prediction, mot_result)
    elif cfg.mot_relax_mode == "gold":
        relax_prediction_with_mot(phrase_grounding_prediction, gold_annotation.images)

    if cfg.coref_relax_mode is not None and cfg.mot_relax_mode is not None:
        # prev_phrase_grounding_prediction = None
        count = 0
        while count < 3:
            # prev_phrase_grounding_prediction = copy.deepcopy(phrase_grounding_prediction)
            if cfg.coref_relax_mode == "pred":
                relax_prediction_with_coreference(phrase_grounding_prediction, parsed_document)
            elif cfg.coref_relax_mode == "gold":
                relax_prediction_with_coreference(phrase_grounding_prediction, gold_document)
            if cfg.mot_relax_mode == "pred":
                mot_result = DetectionLabels.from_json(mot_file.read_text())
                relax_prediction_with_mot(phrase_grounding_prediction, mot_result)
            elif cfg.mot_relax_mode == "gold":
                relax_prediction_with_mot(phrase_grounding_prediction, gold_annotation.images)
            count += 1

    # sort relations by confidence
    for utterance in mm_reference_prediction.utterances:
        for phrase in utterance.phrases:
            phrase.relations.sort(key=lambda rel: rel.bounding_box.confidence, reverse=True)
    prediction_dir.joinpath(f"{parsed_document.did}.json").write_text(
        mm_reference_prediction.to_json(ensure_ascii=False, indent=2)
    )


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


def run_mdetr(
    cfg: DictConfig, dataset_info: DatasetInfo, dataset_dir: Path, document: Document
) -> PhraseGroundingPrediction:
    utterance_results: list[UtterancePrediction] = []
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
        phrases: list[PhrasePrediction] = [
            PhrasePrediction(
                sid=base_phrase.sentence.sid,
                index=base_phrase.global_index,
                text=base_phrase.text,
                relations=[],
            )
            for base_phrase in caption.base_phrases
        ]
        with tempfile.TemporaryDirectory() as out_dir:
            caption_file = Path(out_dir).joinpath("caption.jpp")
            caption_file.write_text(caption.to_jumanpp())
            subprocess.run(
                [
                    cfg.python,
                    f"{cfg.project_root}/run_mdetr.py",
                    f"--model={cfg.checkpoint}",
                    f"--caption-file={caption_file}",
                    f"--backbone-name={cfg.backbone_name}",
                    f"--text-encoder={cfg.text_encoder}",
                    f"--batch-size={cfg.batch_size}",
                    f"--export-dir={out_dir}",
                    "--image-files",
                ]
                + [str(dataset_dir / image.path) for image in corresponding_images],
                check=True,
            )
            predictions = [MDETRPrediction.from_json(file.read_text()) for file in sorted(Path(out_dir).glob("*.json"))]

        assert len(corresponding_images) == len(predictions), f"{len(corresponding_images)} != {len(predictions)}"
        for (image, prediction), (phrase, base_phrase) in itertools.product(
            zip(corresponding_images, predictions),
            zip(phrases, caption.base_phrases),
        ):
            assert prediction.image_id == image.id
            for bounding_box in prediction.bounding_boxes:
                prob = max(bounding_box.word_probs[m.global_index] for m in base_phrase.morphemes)
                if prob >= 0.1:
                    phrase.relations.append(
                        RelationPrediction(
                            type="=",
                            image_id=image.id,
                            bounding_box=BoundingBoxPrediction(
                                image_id=image.id,
                                rect=bounding_box.rect,
                                confidence=bounding_box.confidence,
                            ),
                        )
                    )
        for phrase in phrases:
            phrase.relations.sort(key=lambda rel: rel.bounding_box.confidence, reverse=True)
        utterance_results.append(UtterancePrediction(text=caption.text, sids=utterance.sids, phrases=phrases))

    return PhraseGroundingPrediction(
        scenario_id=dataset_info.scenario_id, images=dataset_info.images, utterances=utterance_results
    )


def run_glip(
    cfg: DictConfig, dataset_info: DatasetInfo, dataset_dir: Path, document: Document
) -> PhraseGroundingPrediction:
    utterance_predictions: list[UtterancePrediction] = []
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
        phrases: list[PhrasePrediction] = [
            PhrasePrediction(
                sid=base_phrase.sentence.sid,
                index=base_phrase.global_index,
                text=base_phrase.text,
                relations=[],
            )
            for base_phrase in caption.base_phrases
        ]
        with tempfile.TemporaryDirectory() as out_dir:
            caption_file = Path(out_dir).joinpath("caption.knp")
            caption_file.write_text(caption.to_knp())
            subprocess.run(
                [
                    cfg.python,
                    f"{cfg.project_root}/tools/run_glip.py",
                    f"--model={cfg.checkpoint}",
                    f"--config-file={cfg.config}",
                    f"--caption-file={caption_file}",
                    f"--export-dir={out_dir}",
                    "--image-files",
                ]
                + [str(dataset_dir / image.path) for image in corresponding_images],
                check=True,
            )
            predictions = [GLIPPrediction.from_json(file.read_text()) for file in sorted(Path(out_dir).glob("*.json"))]

        assert len(corresponding_images) == len(predictions), f"{len(corresponding_images)} != {len(predictions)}"
        for image, prediction in zip(corresponding_images, predictions):
            for phrase, phrase_prediction in zip(phrases, prediction.phrases):
                assert phrase_prediction.text == phrase.text
                assert phrase_prediction.phrase_index == phrase.index
                for bounding_box in phrase_prediction.bounding_boxes:
                    assert bounding_box.image_id == image.id
                    phrase.relations.append(
                        RelationPrediction(
                            type="=",
                            image_id=image.id,
                            bounding_box=BoundingBoxPrediction(
                                image_id=image.id,
                                rect=bounding_box.rect,
                                confidence=bounding_box.confidence,
                            ),
                        )
                    )
        utterance_predictions.append(UtterancePrediction(text=caption.text, sids=utterance.sids, phrases=phrases))

    return PhraseGroundingPrediction(
        scenario_id=dataset_info.scenario_id, images=dataset_info.images, utterances=utterance_predictions
    )


def relax_prediction_with_mot(
    phrase_grounding_prediction: PhraseGroundingPrediction,
    image_annotations: Union[list[ImageAnnotation], DetectionLabels],
    confidence_modification_method: str = "max",  # min, max, mean
) -> None:
    # create a bounding box cluster according to instance_id
    gold_bb_cluster: dict[str, list[BoundingBoxAnnotation]] = defaultdict(list)
    if isinstance(image_annotations, DetectionLabels):
        for idx in range(math.ceil(len(image_annotations.frames) / 30)):
            frame = image_annotations.frames[idx * 30]  # TODO: identify exact frame
            for bb in frame.bounding_boxes:
                gold_bb_cluster[str(bb.instance_id)].append(
                    BoundingBoxAnnotation(
                        image_id=f"{idx:03d}",
                        rect=bb.rect,
                        class_name=bb.class_name,
                        instance_id=str(bb.instance_id),
                    )
                )
    else:
        for image_annotation in image_annotations:
            for bb in image_annotation.bounding_boxes:
                gold_bb_cluster[bb.instance_id].append(bb)

    phrase_predictions = [pp for utterance in phrase_grounding_prediction.utterances for pp in utterance.phrases]
    for phrase_prediction in phrase_predictions:
        for gold_bbs in gold_bb_cluster.values():
            # このクラスタに属する relation を集める
            relations_in_cluster: list[RelationPrediction] = []
            for gold_bb in gold_bbs:
                for relation in phrase_prediction.relations:
                    if relation.type != "=":
                        continue
                    if (
                        relation.image_id == gold_bb.image_id
                        and box_iou(relation.bounding_box.rect, gold_bb.rect) >= 0.5
                    ):
                        relations_in_cluster.append(relation)
            if len(relations_in_cluster) == 0:
                continue
            confidences = [rel.bounding_box.confidence for rel in relations_in_cluster]
            if confidence_modification_method == "max":
                modified_confidence = max(confidences)
            elif confidence_modification_method == "min":
                modified_confidence = min(confidences)
            elif confidence_modification_method == "mean":
                modified_confidence = mean(confidences)
            else:
                raise NotImplementedError
            print([f"{conf:.6f}" for conf in confidences])
            for rel in relations_in_cluster:
                print(
                    f"{phrase_grounding_prediction.scenario_id}: {rel.image_id}: {gold_bbs[0].class_name}: confidence: {rel.bounding_box.confidence:.6f} -> {modified_confidence:.6f}"
                )
                rel.bounding_box.confidence = modified_confidence


def relax_prediction_with_coreference(
    phrase_grounding_prediction: PhraseGroundingPrediction, document: Document
) -> None:
    sid2sentence: dict[str, Sentence] = {sentence.sid: sentence for sentence in document.sentences}

    phrase_id_to_relations: dict[int, set[RelationPrediction]] = defaultdict(set)
    # convert phrase grounding result to phrase_id_to_relations
    for utterance in phrase_grounding_prediction.utterances:
        base_phrases = [bp for sid in utterance.sids for bp in sid2sentence[sid].base_phrases]
        for base_phrase, phrase_prediction in zip(base_phrases, utterance.phrases):
            phrase_id_to_relations[base_phrase.global_index].update(phrase_prediction.relations)

    for utterance in phrase_grounding_prediction.utterances:
        base_phrases = [bp for sid in utterance.sids for bp in sid2sentence[sid].base_phrases]
        for base_phrase, phrase_prediction in zip(base_phrases, utterance.phrases):
            coreferents: list[BasePhrase] = base_phrase.get_coreferents(include_nonidentical=False)
            relations = set(phrase_prediction.relations)
            for coreferent_base_phrase in coreferents:
                relations.update(phrase_id_to_relations[coreferent_base_phrase.global_index])
            phrase_prediction.relations = list(relations)


def relax_prediction_with_pas_bridging(
    phrase_grounding_prediction: PhraseGroundingPrediction,
    parsed_document: Document,
) -> PhraseGroundingPrediction:
    phrase_id_to_relations: dict[int, set[RelationPrediction]] = defaultdict(set)

    # convert phrase grounding result to phrase_id_to_relations
    sid2sentence = {sentence.sid: sentence for sentence in parsed_document.sentences}
    for utterance in phrase_grounding_prediction.utterances:
        base_phrases = [bp for sid in utterance.sids for bp in sid2sentence[sid].base_phrases]
        for base_phrase, phrase_prediction in zip(base_phrases, utterance.phrases):
            phrase_id_to_relations[base_phrase.global_index].update(phrase_prediction.relations)

    # relax annotation until convergence
    phrase_id_to_relations_prev: dict[int, set[RelationPrediction]] = {}
    while phrase_id_to_relations != phrase_id_to_relations_prev:
        phrase_id_to_relations_prev = copy.deepcopy(phrase_id_to_relations)
        relax_annotation_with_pas_bridging(parsed_document, phrase_id_to_relations)

    # convert phrase_id_to_relations to phrase grounding result
    for utterance in phrase_grounding_prediction.utterances:
        base_phrases = [bp for sid in utterance.sids for bp in sid2sentence[sid].base_phrases]
        for base_phrase, phrase_prediction in zip(base_phrases, utterance.phrases):
            relations = set(phrase_prediction.relations)
            relations.update(phrase_id_to_relations[base_phrase.global_index])
            phrase_prediction.relations = list(relations)

    return phrase_grounding_prediction


def relax_annotation_with_pas_bridging(
    document: Document, phrase_id_to_relations: dict[int, set[RelationPrediction]]
) -> None:
    for base_phrase in document.base_phrases:
        current_relations: set[RelationPrediction] = set()  # 述語を中心とした関係の集合
        current_relations.update(phrase_id_to_relations[base_phrase.global_index])
        new_relations: set[RelationPrediction] = set([])
        for case, arguments in base_phrase.pas.get_all_arguments(relax=False).items():
            argument_global_indices: set[int] = set()
            for argument in arguments:
                if isinstance(argument, EndophoraArgument):
                    argument_global_indices.add(argument.base_phrase.global_index)
            new_relations.update(
                {
                    RelationPrediction(type=case, image_id=rel.image_id, bounding_box=rel.bounding_box)
                    for argument_global_index in argument_global_indices
                    for rel in phrase_id_to_relations[argument_global_index]
                    if rel.type == "="
                }
            )
            # 格が一致する述語を中心とした関係の集合
            # relation の対象は argument_entity_ids と一致
            case_relations: set[RelationPrediction] = {rel for rel in current_relations if rel.type == case}
            for argument_global_index in argument_global_indices:
                phrase_id_to_relations[argument_global_index].update(
                    {
                        RelationPrediction(type="=", image_id=rel.image_id, bounding_box=rel.bounding_box)
                        for rel in case_relations
                    }
                )
        phrase_id_to_relations[base_phrase.global_index].update(new_relations)


def preprocess_document(document: Document) -> Document:
    for base_phrase in document.base_phrases:
        filtered = RelTagList()
        for rel_tag in base_phrase.rel_tags:
            # exclude '?' rel tags for simplicity
            if rel_tag.mode is RelMode.AMBIGUOUS and rel_tag.target != "なし":
                continue
            # exclude coreference relations of 用言
            # e.g., ...を[運んで]。[それ]が終わったら...
            if rel_tag.type == "=" and rel_tag.sid is not None:
                if target_base_phrase := base_phrase._get_target_base_phrase(rel_tag):
                    if ("体言" in base_phrase.features and "体言" in target_base_phrase.features) is False:
                        continue
            filtered.append(rel_tag)
        base_phrase.rel_tags = filtered
    document = document.reparse()
    # ensure that each base phrase has at least one entity
    for base_phrase in document.base_phrases:
        if len(base_phrase.entities) == 0:
            EntityManager.get_or_create_entity().add_mention(base_phrase)
    return document


if __name__ == "__main__":
    main()
