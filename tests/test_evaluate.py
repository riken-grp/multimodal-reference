from pathlib import Path

from rhoknp import Document

from evaluation import MMRefEvaluator
from prediction_writer import PhraseGroundingResult
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
"""


def test_evaluate(fixture_data_dir: Path):
    evaluate_dir = fixture_data_dir / 'evaluate'
    image_text_annotation = ImageTextAnnotation.from_json(evaluate_dir.joinpath('converted.json').read_text())
    evaluator = MMRefEvaluator(
        DatasetInfo.from_json(evaluate_dir.joinpath('info.json').read_text()),
        Document.from_knp(evaluate_dir.joinpath('reference.knp').read_text()),
        image_text_annotation,
    )

    prediction = PhraseGroundingResult.from_json(evaluate_dir.joinpath('prediction.json').read_text())

    result: dict[str, dict[str, float]] = evaluator.eval_visual_reference(prediction)
    assert result['ガ'] == {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    assert result['ヲ'] == {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    assert result['ニ'] == {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    assert result['ノ'] == {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    assert result['='] == {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
