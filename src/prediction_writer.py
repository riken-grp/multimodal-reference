import copy
import math
import warnings
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Annotated, Optional, Union

import hydra
import luigi
from omegaconf import DictConfig
from rhoknp import BasePhrase, Document, Sentence
from rhoknp.cohesion import EndophoraArgument, RelMode, RelTagList
from rhoknp.cohesion.coreference import EntityManager

from tasks import (
    CohesionAnalysis,
    DeticPhraseGrounding,
    GLIPPhraseGrounding,
    MDETRPhraseGrounding,
    MultipleObjectTracking,
)
from utils.annotation import BoundingBox as BoundingBoxAnnotation
from utils.annotation import ImageAnnotation, ImageTextAnnotation
from utils.mot import DetectionLabels
from utils.prediction import PhraseGroundingPrediction, RelationPrediction
from utils.util import box_iou

warnings.filterwarnings(
    "ignore",
    message=r'Parameter "(cfg|document)" with value .+ is not of type string\.',
    category=UserWarning,
)


@hydra.main(version_base=None, config_path="../configs")
def main(cfg: DictConfig) -> None:
    if len(cfg.scenario_ids) == 0 and cfg.id_file is not None:
        cfg.scenario_ids = Path(cfg.id_file).read_text().strip().split()
    tasks: list[luigi.Task] = []
    for scenario_id in cfg.scenario_ids:
        tasks.append(MultimodalReference(cfg=cfg, scenario_id=scenario_id))
    luigi.build(tasks, **cfg.luigi)


class MultimodalReference(luigi.Task):
    cfg: Annotated[DictConfig, luigi.Parameter()] = luigi.Parameter()
    scenario_id: Annotated[str, luigi.Parameter()] = luigi.Parameter()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dataset_dir = Path(self.cfg.dataset_dir) / self.scenario_id
        self.exp_dir = Path(self.cfg.exp_dir)
        self.gold_document = Document.from_knp(
            Path(self.cfg.gold_knp_dir).joinpath(f"{self.scenario_id}.knp").read_text()
        )

    def requires(self) -> dict[str, luigi.Task]:
        tasks: dict[str, luigi.Task] = {
            "cohesion": CohesionAnalysis(cfg=self.cfg.cohesion, scenario_id=self.scenario_id)
        }
        if self.cfg.phrase_grounding_model == "glip":
            tasks["grounding"] = GLIPPhraseGrounding(
                cfg=self.cfg.glip,
                scenario_id=self.scenario_id,
                document=self.gold_document,
                dataset_dir=self.dataset_dir,
            )
        elif self.cfg.phrase_grounding_model == "mdetr":
            tasks["grounding"] = MDETRPhraseGrounding(
                cfg=self.cfg.mdetr,
                scenario_id=self.scenario_id,
                document=self.gold_document,
                dataset_dir=self.dataset_dir,
            )
        else:
            assert self.cfg.phrase_grounding_model == "detic"
            tasks["grounding"] = DeticPhraseGrounding(
                cfg=self.cfg.detic,
                scenario_id=self.scenario_id,
                document=self.gold_document,
                dataset_dir=self.dataset_dir,
            )
        if self.cfg.mot_relax_mode == "pred":
            tasks["mot"] = MultipleObjectTracking(cfg=self.cfg.mot, scenario_id=self.scenario_id)
        return tasks

    def output(self):
        return luigi.LocalTarget(self.exp_dir.joinpath(f"{self.scenario_id}.json"))

    def run(self):
        with self.input()["cohesion"].open(mode="r") as f:
            cohesion_prediction = Document.from_knp(f.read())
        with self.input()["grounding"].open(mode="r") as f:
            phrase_grounding_prediction = PhraseGroundingPrediction.from_json(f.read())
        mot_prediction: Optional[DetectionLabels] = None
        if self.cfg.mot_relax_mode == "pred":
            with self.input()["mot"].open(mode="r") as f:
                mot_prediction = DetectionLabels.from_json(f.read())
        gold_annotation = ImageTextAnnotation.from_json(
            Path(self.cfg.gold_annotation_dir).joinpath(f"{self.scenario_id}.json").read_text()
        )
        prediction = run_prediction(
            cfg=self.cfg,
            cohesion_prediction=cohesion_prediction,
            phrase_grounding_prediction=phrase_grounding_prediction,
            mot_prediction=mot_prediction,
            gold_document=self.gold_document,
            image_annotations=gold_annotation.images,
        )
        with self.output().open(mode="w") as f:
            f.write(prediction.to_json(ensure_ascii=False, indent=2))


def run_prediction(
    cfg: DictConfig,
    cohesion_prediction: Document,
    phrase_grounding_prediction: PhraseGroundingPrediction,
    mot_prediction: Optional[DetectionLabels],
    gold_document: Document,
    image_annotations: list[ImageAnnotation],
) -> PhraseGroundingPrediction:
    parsed_document = preprocess_document(cohesion_prediction)

    if cfg.coref_relax_mode == "pred":
        relax_prediction_with_coreference(phrase_grounding_prediction, parsed_document)
    elif cfg.coref_relax_mode == "gold":
        relax_prediction_with_coreference(phrase_grounding_prediction, gold_document)

    if cfg.mot_relax_mode == "pred":
        assert mot_prediction is not None
        relax_prediction_with_mot(phrase_grounding_prediction, mot_prediction)
    elif cfg.mot_relax_mode == "gold":
        relax_prediction_with_mot(phrase_grounding_prediction, image_annotations)

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
                assert mot_prediction is not None
                relax_prediction_with_mot(phrase_grounding_prediction, mot_prediction)
            elif cfg.mot_relax_mode == "gold":
                relax_prediction_with_mot(phrase_grounding_prediction, image_annotations)
            count += 1

    mm_reference_prediction = relax_prediction_with_pas_bridging(phrase_grounding_prediction, parsed_document)

    # sort relations by confidence
    for utterance in mm_reference_prediction.utterances:
        for phrase in utterance.phrases:
            phrase.relations.sort(key=lambda rel: rel.bounding_box.confidence, reverse=True)
    return mm_reference_prediction


def relax_prediction_with_mot(
    phrase_grounding_prediction: PhraseGroundingPrediction,
    image_annotations: Union[list[ImageAnnotation], DetectionLabels],
    confidence_modification_method: str = "max",  # min, max, mean
) -> None:
    # create a bounding box cluster according to instance_id
    gold_bb_cluster: dict[str, list[BoundingBoxAnnotation]] = defaultdict(list)
    if isinstance(image_annotations, DetectionLabels):
        for idx in range(math.ceil(len(image_annotations.frames) / 30)):
            frame = image_annotations.frames[idx * 30]
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
            for gold_bb in image_annotation.bounding_boxes:
                gold_bb_cluster[gold_bb.instance_id].append(gold_bb)

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
