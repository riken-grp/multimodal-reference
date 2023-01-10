# Multi-modal Reference Resolution

## Setup Environment

```shell
poetry install
```

## Run SeparateModel

```shell
poetry run ./run_separate_model.sh
```

```text
dataset/<dialogue-id>
- images/
- info.json
- raw.knp
- tracking.json
```

```text
result/<dialogue-id>
- images/
- info.json
- raw.knp
- tracking.json
```

## 実装方針

入力: 対話テキスト, 画像, タイムスタンプ -> dataset そのもの？
出力: KNP ファイル, BB（coord, instance id, class）, MDETR出力
- prediction directory を用意する
評価用入力: gold KNP ファイル, Image-Text Annotation ファイル
- evaluator は dataset と prediction directory を読み込む

```shell
poetry run python src/evaluation.py --dataset-dir data/dataset --gold-knp-dir data/knp --gold-image-dir data/image_text_annotation --prediction-dir data/prediction --scenario-ids 20220302-56130229-0 --result-json result.json
```

## Evaluation Result

```json
{
  "scenarioId": "<scenario-id>",
  "phrases": [
    {
      "sid": "<sentence-id>",
      "phraseId": "<phrase-id>",
      "text": "<phrase-text>",
      "relation": "=",
      "image": "image.png",
      "boundingBoxes": [
        {
          "imageId": "<image-id>",
          "bb": [
            0,
            0,
            0,
            0
          ],
          "className": "<class-name>"
        }
      ]
    }
  ]
}
```
