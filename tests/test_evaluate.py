from pathlib import Path

from rhoknp import Document

from evaluation import MMRefEvaluator
from prediction_writer import PhraseGroundingPrediction
from utils.image import ImageTextAnnotation
from utils.util import DatasetInfo

"""
画像3枚
2発話
- 風船の下にギターがある。隣には段ボール箱がある。
- 風船の下にギターがある。隣には段ボール箱がある。
- 上に移動した。
reference is 何？
<<<風船>>>の<<<下>>>に<<<ギター>>>が<<<ある>>>。<<<隣>>>には段ボール<<<箱>>>が<<<ある>>>。
<<<上>>>に<<<移動した>>>。

tp/denom_gold/denom_pred
1枚目=格
風船: 0/1/3
ギター: 0/1/0
段ボール: 0/0/5
箱: 2/2/2 (2/2/5 のうち pred 側の重複は除去 ( = gold bb 1つにつき pred bb 1つ))

2枚目=格
風船: 0/1/3
ギター: 0/1/0
段ボール: 0/0/8
箱: 2/2/4 (2/2/8 のうち pred 側の重複は除去)

3枚目=格
上: 0/1/0

1枚目ガ格
ある: -/1/-
ある: -/2/-

2枚目ガ格
ある: -/1/-
ある: -/2/-

3枚目=格
移動した: -/1/-
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

    result: dict[str, dict] = evaluator.eval_visual_reference(prediction)
    assert (result['ガ']['recall_pos'], result['ガ']['recall_total'], result['ガ']['precision_pos']) == (-1, 7, -1)
    # assert result['ヲ'] == {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    # assert result['ニ'] == {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    # assert result['ノ'] == {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    assert (result['=']['recall_pos'], result['=']['recall_total'], result['=']['precision_pos']) == (4, 9, 25)
