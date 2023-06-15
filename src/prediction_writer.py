import copy
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import hydra
from omegaconf import DictConfig
from PIL import Image, ImageFile
from rhoknp import Document
from rhoknp.cohesion import EndophoraArgument, ExophoraArgument, RelMode, RelTagList
from rhoknp.cohesion.coreference import EntityManager

from mdetr import BoundingBox, MDETRPrediction, predict_mdetr
from utils.util import CamelCaseDataClassJsonMixin, DatasetInfo, ImageInfo


@dataclass(frozen=True, eq=True)
class RelationPrediction(CamelCaseDataClassJsonMixin):
    type: str  # ガ, ヲ, ニ, ノ, =, etc...
    image_id: str
    bounding_box: BoundingBox


@dataclass
class PhrasePrediction(CamelCaseDataClassJsonMixin):
    sid: str
    index: int
    text: str
    relations: list[RelationPrediction]


@dataclass
class UtterancePrediction(CamelCaseDataClassJsonMixin):
    text: str
    sids: list[str]
    phrases: list[PhrasePrediction]


@dataclass
class PhraseGroundingPrediction(CamelCaseDataClassJsonMixin):
    scenario_id: str
    images: list[ImageInfo]
    utterances: list[UtterancePrediction]


@hydra.main(version_base=None, config_path="../configs")
def main(cfg: DictConfig) -> None:
    dataset_dir = Path(cfg.dataset_dir)
    gold_knp_file = Path(cfg.gold_knp_file)  # TODO: remove gold tags just in case
    prediction_dir = Path(cfg.prediction_dir)
    prediction_dir.mkdir(exist_ok=True)

    dataset_info = DatasetInfo.from_json(dataset_dir.joinpath('info.json').read_text())

    pred_knp_file = prediction_dir / f'{dataset_info.scenario_id}.knp'
    if pred_knp_file.exists():
        parsed_document = Document.from_knp(pred_knp_file.read_text())
    else:
        parsed_document = run_cohesion(cfg.cohesion, gold_knp_file)
        pred_knp_file.write_text(parsed_document.to_knp())

    # perform phrase grounding with MDETR
    phrase_grounding_file = prediction_dir / 'mdetr' / f'{parsed_document.did}.json'
    phrase_grounding_file.parent.mkdir(exist_ok=True)
    if phrase_grounding_file.exists():
        mdetr_result = PhraseGroundingPrediction.from_json(phrase_grounding_file.read_text())
    else:
        mdetr_result = run_mdetr(cfg.mdetr, dataset_info, dataset_dir, parsed_document)
        phrase_grounding_file.write_text(mdetr_result.to_json(ensure_ascii=False, indent=2))

    parsed_document = preprocess_document(parsed_document)
    mm_reference_prediction = relax_prediction(mdetr_result, parsed_document)
    prediction_dir.joinpath(f'{parsed_document.did}.json').write_text(
        mm_reference_prediction.to_json(ensure_ascii=False, indent=2)
    )


def run_cohesion(cfg: DictConfig, input_knp_file: Path) -> Document:
    with tempfile.TemporaryDirectory() as out_dir:
        subprocess.run(
            [
                cfg.python,
                f'{cfg.project_root}/src/predict.py',
                f'checkpoint={cfg.checkpoint}',
                f'input_path={input_knp_file}',
                f'export_dir={out_dir}',
                'num_workers=0',
            ]
        )
        return Document.from_knp(next(Path(out_dir).glob('*.knp')).read_text())


def run_mdetr(
    cfg: DictConfig, dataset_info: DatasetInfo, dataset_dir: Path, document: Document
) -> PhraseGroundingPrediction:
    utterance_results: list[UtterancePrediction] = []
    sid2sentence = {sentence.sid: sentence for sentence in document.sentences}
    for utterance in dataset_info.utterances:
        corresponding_images = [image for image in dataset_info.images if image.id in utterance.image_ids]
        caption = Document.from_sentences([sid2sentence[sid] for sid in utterance.sids])
        phrases: list[PhrasePrediction] = [
            PhrasePrediction(
                sid=base_phrase.sentence.sid,
                index=base_phrase.index,
                text=base_phrase.text,
                relations=[],
            )
            for base_phrase in caption.base_phrases
        ]
        image_files: list[ImageFile] = [Image.open(dataset_dir.joinpath(image.path)) for image in corresponding_images]
        predictions: list[MDETRPrediction] = predict_mdetr(cfg.checkpoint, image_files, caption, cfg.batch_size)
        for image, prediction in zip(corresponding_images, predictions):
            for bounding_box in prediction.bounding_boxes:
                for phrase, base_phrase in zip(phrases, caption.base_phrases):
                    words = [prediction.words[m.global_index] for m in base_phrase.morphemes]
                    assert ''.join(words) == phrase.text == base_phrase.text
                    prob = max(bounding_box.word_probs[m.global_index] for m in base_phrase.morphemes)
                    if prob >= 0.1:
                        phrase.relations.append(
                            RelationPrediction(type='=', image_id=image.id, bounding_box=bounding_box)
                        )
        utterance_results.append(UtterancePrediction(text=caption.text, sids=utterance.sids, phrases=phrases))

    return PhraseGroundingPrediction(
        scenario_id=dataset_info.scenario_id, images=dataset_info.images, utterances=utterance_results
    )


def relax_prediction(
    phrase_grounding_result: PhraseGroundingPrediction,
    parsed_document: Document,
) -> PhraseGroundingPrediction:
    eid2relations: dict[int, set[RelationPrediction]] = defaultdict(set)

    # convert phrase grounding result to eid2relations
    sid2sentence = {sentence.sid: sentence for sentence in parsed_document.sentences}
    for utterance in phrase_grounding_result.utterances:
        base_phrases = [bp for sid in utterance.sids for bp in sid2sentence[sid].base_phrases]
        for base_phrase, phrase_prediction in zip(base_phrases, utterance.phrases):
            for entity in base_phrase.entities:
                eid2relations[entity.eid].update(phrase_prediction.relations)

    # relax annotation until convergence
    eid2relations_prev: dict[int, set[RelationPrediction]] = {}
    while eid2relations != eid2relations_prev:
        eid2relations_prev = copy.deepcopy(eid2relations)
        relax_annotation(parsed_document, eid2relations)

    # convert eid2relations to phrase grounding result
    for utterance in phrase_grounding_result.utterances:
        base_phrases = [bp for sid in utterance.sids for bp in sid2sentence[sid].base_phrases]
        for base_phrase, phrase_prediction in zip(base_phrases, utterance.phrases):
            relations = set(phrase_prediction.relations)
            for entity in base_phrase.entities:
                relations.update(eid2relations[entity.eid])
            phrase_prediction.relations = list(relations)

    return phrase_grounding_result


def relax_annotation(document: Document, eid2relations: dict[int, set[RelationPrediction]]) -> None:
    for base_phrase in document.base_phrases:
        current_relations: set[RelationPrediction] = set()  # 述語を中心とした関係の集合
        for entity in base_phrase.entities:
            current_relations.update(eid2relations[entity.eid])
        new_relations: set[RelationPrediction] = set([])
        pas = base_phrase.pas
        for case, arguments in pas.get_all_arguments(relax=False).items():
            argument_entity_ids: set[int] = set()
            for argument in arguments:
                if isinstance(argument, EndophoraArgument):
                    argument_entity_ids.update({entity.eid for entity in argument.base_phrase.entities})
                elif isinstance(argument, ExophoraArgument):
                    argument_entity_ids.add(argument.eid)
                else:
                    raise AssertionError
            new_relations.update(
                {
                    RelationPrediction(type=case, image_id=rel.image_id, bounding_box=rel.bounding_box)
                    for argument_eid in argument_entity_ids
                    for rel in eid2relations[argument_eid]
                    if rel.type == '='
                }
            )
            # 格が一致する述語を中心とした関係の集合
            # relation の対象は argument_entity_ids と一致
            case_relations: set[RelationPrediction] = {rel for rel in current_relations if rel.type == case}
            for argument_eid in argument_entity_ids:
                eid2relations[argument_eid].update(
                    {
                        RelationPrediction(type='=', image_id=rel.image_id, bounding_box=rel.bounding_box)
                        for rel in case_relations
                    }
                )
        for entity in base_phrase.entities:
            eid2relations[entity.eid].update(new_relations)


def preprocess_document(document: Document) -> Document:
    for base_phrase in document.base_phrases:
        filtered = RelTagList()
        for rel_tag in base_phrase.rel_tags:
            # exclude '?' rel tags for simplicity
            if rel_tag.mode is RelMode.AMBIGUOUS and rel_tag.target != 'なし':
                continue
            # exclude coreference relations of 用言
            # e.g., ...を[運んで]。[それ]が終わったら...
            if rel_tag.type == '=' and rel_tag.sid is not None:
                if target_base_phrase := base_phrase._get_target_base_phrase(rel_tag):
                    if ('体言' in base_phrase.features and '体言' in target_base_phrase.features) is False:
                        continue
            filtered.append(rel_tag)
        base_phrase.rel_tags = filtered
    document = document.reparse()
    # ensure that each base phrase has at least one entity
    for base_phrase in document.base_phrases:
        if len(base_phrase.entities) == 0:
            EntityManager.get_or_create_entity().add_mention(base_phrase)
    return document


if __name__ == '__main__':
    main()
