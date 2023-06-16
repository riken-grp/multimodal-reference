from pathlib import Path
from typing import Any

import polars as pl
from rhoknp import Document

from evaluation import MMRefEvaluator
from prediction_writer import PhraseGroundingPrediction
from utils.image import ImageTextAnnotation
from utils.util import DatasetInfo

"""
画像3枚: 001.png, 002.png, 003.png
2発話
- 風船の下にギターがある。隣には段ボール箱がある。
  - 001.png
  - 002.png
- 上に移動した。
  - 003.png

<<<風船>>>の<<<下>>>に<<<ギター>>>が<<<ある>>>。<<<隣>>>には段ボール<<<箱>>>が<<<ある>>>。
<<<上>>>に<<<移動した>>>。

tp/recall_total/precision_total
- =格
  - 001
    風船: 0/1/3
    ギター: 0/1/0
    段ボール: 0/0/5
    箱: 2/2/2 (2/2/5 のうち pred 側の重複は除去 ( = gold bb 1つにつき pred bb 1つ))
    total: 2/4/13

  - 002
    風船: 0/1/3
    ギター: 0/1/0
    段ボール: 0/0/8
    箱: 2/2/4 (2/2/8 のうち pred 側の重複は除去)
    total: 2/4/19

  - 003
    上: 0/1/0
    total: 0/1/0

- ガ格
  - 001
    ある: 0/1/0
    ある: 2/2/5
    total: 2/3/5

  - 002
    ある: 0/1/0
    ある: 2/2/8
    total: 2/3/8

  - 003
    移動した: 0/1/0
    total: 0/1/0

- ヲ格
  - 003
    移動した: 0/0/0
    total: 0/0/0

- ノ格
  - 001
    下: 0/1/3
    隣: 0/1/3
    total: 0/2/6

  - 002
    下: 0/1/3
    隣: 0/1/3
    total: 0/2/6

  - 003
    上: 0/1/0
    total: 0/1/0
"""


def test_evaluate(fixture_data_dir: Path):
    evaluate_dir = fixture_data_dir / 'evaluate'
    image_text_annotation = ImageTextAnnotation.from_json(evaluate_dir.joinpath('gold.json').read_text())
    evaluator = MMRefEvaluator(
        DatasetInfo.from_json(evaluate_dir.joinpath('info.json').read_text()),
        Document.from_knp(evaluate_dir.joinpath('gold.knp').read_text()),
        image_text_annotation,
    )

    prediction = PhraseGroundingPrediction.from_json(
        evaluate_dir.joinpath(f'{image_text_annotation.scenario_id}.json').read_text()
    )

    results: list[dict[str, Any]] = evaluator.eval_visual_reference(prediction)
    df = pl.DataFrame(results)
    df_rel = df.groupby('relation_type', maintain_order=True).sum()
    result = {}
    for rel in ('ガ', 'ヲ', 'ニ', 'ノ', '='):
        result[rel] = {
            k: 0 if v.is_empty() else v.item()
            for k, v in df_rel.filter(pl.col('relation_type') == rel)
            .select(pl.col('recall_pos', 'recall_total', 'precision_pos', 'precision_total'))
            .to_dict()
            .items()
        }
    assert (result['ガ']['recall_pos'], result['ガ']['recall_total'], result['ガ']['precision_total']) == (4, 7, 13)
    assert (result['ヲ']['recall_pos'], result['ヲ']['recall_total'], result['ヲ']['precision_total']) == (0, 0, 0)
    assert (result['ニ']['recall_pos'], result['ニ']['recall_total'], result['ニ']['precision_total']) == (0, 1, 0)
    assert (result['ノ']['recall_pos'], result['ノ']['recall_total'], result['ノ']['precision_total']) == (0, 4, 12)
    assert (result['=']['recall_pos'], result['=']['recall_total'], result['=']['precision_total']) == (4, 9, 32)
