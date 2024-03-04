# Multi-modal Reference Resolution

## Requirements

- Python: >=3.9,<3.12
- Python dependencies: See [pyproject.toml](./pyproject.toml).
- [nobu-g/GLIP](https://github.com/nobu-g/GLIP)
- [nobu-g/cohesion-analysis](https://github.com/nobu-g/cohesion-analysis)

## Setup Environment

```shell
poetry env use /path/to/python
poetry install
```

## Prepare Datasets

1. Download the J-CRe3 annotations.

    ```shell
    git clone git@github.com:riken-grp/J-CRe3.git /somewhere/J-CRe3
    ```

2. Download the video and audio files following [the instructions](https://github.com/riken-grp/J-CRe3?tab=readme-ov-file#video-and-audio-files) in J-CRe3.

3. Place data files under `data` directory. You can use `cp -r` instead of `ln -s`.

    ```shell
    mkdir -p data
    ln -s /somewhere/J-CRe3/textual_annotations ./data/knp
    ln -s /somewhere/J-CRe3/visual_annotations ./data/image_text_annotation
    ln -s /somewhere/J-CRe3/id ./data/id
    ln -s /somewhere/J-CRe3/recording ./data/recording
    ln -s /somewhere/J-CRe3/recording ./data/dataset
   ```

## Setup GLIP

1. Follow the instructions in [nobu-g/GLIP](https://github.com/nobu-g/GLIP) to set up the GLIP environment. We recommend to use Poetry to install the dependencies.

2. Download the checkpoint files under the GLIP root directory.

    ```shell
    cd /somewhere/GLIP
    wget https://lotus.kuee.kyoto-u.ac.jp/~ueda/dist/GLIP/OUTPUT.zip
    unzip OUTPUT.zip
    ```

## Setup cohesion analyzer

TBW

## Run Prediction

1. Make a config file for cohesion analysis.

  - Copy the example config file and modify it as needed.

   ```shell
   cp configs/cohesion/example.yaml configs/cohesion/default.yaml
   ```
   - Modify the `python`, `project_root`, and `checkpoint` fields in `configs/cohesion/default.yaml` as needed.

2. Make a config file for phrase grounding by GLIP.

  - Copy the example config file and modify it as needed.

   ```shell
   cp configs/glip/example.yaml configs/glip/ft2_deberta_b24_u3s1_b48_1e.yaml
   ```
   - Modify the `python`, `project_root`, `name`, `exp_name`, and `checkpoint` fields in `configs/glip/ft2_deberta_b24_u3s1_b48_1e.yaml` as needed.
     The `name` field should be the same as the file name of the config file (`ft2_deberta_b24_u3s1_b48_1e`).

3. Make a config file for prediction writer.

  - Copy the example config file and modify it as needed.

   ```shell
   cp configs/example.yaml configs/default.yaml
   ```
   - Modify the `cohesion`, `glip`, and `checkpoint` fields in `configs/prediction_writer/default.yaml` as needed.

4. Run `prediction_writer.py`.

   ```shell
   [AVAILABLE_GPUS=0,1,2,3] python src/prediction_writer.py -cn default \
       phrase_grounding_model=glip \
       glip=ft2_deberta_b24_u3s1_b48_1e \
       id_file=<(cat data/id/test.id data/id/valid.id) \
       luigi.workers=4
    ```

## Run Evaluation

```shell
python src/evaluation.py \
  --dataset-dir data/dataset \
  --gold-knp-dir data/knp \
  --gold-annotation-dir data/image_text_annotation \
  --prediction-mmref-dir result/mmref/glip_ft2_deberta_b24_u3s1_b48_1e \
  --prediction-knp-dir result/cohesion \
  --scenario-ids $(cat data/id/test.id) \
  --recall-topk -1 1 5 10 \
  --confidence-threshold 0.0 \
  --column-prefixes rel_type prec rec
```

## Citation

```bibtex
@inproceedings{ueda-2024-jcre3,
  title={J-CRe3: A Japanese Conversation Dataset for Real-world Reference Resolution},
  author={Nobuhiro Ueda and Hideko Habe and Yoko Matsui and Akishige Yuguchi and Seiya Kawano and Yasutomo Kawanishi and Sadao Kurohashi and Koichiro Yoshino},
  booktitle={Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
  year={2024},
  pages={},
  address={Turin, Italy},
}
```

```bibtex
@inproceedings{植田2023a,
  author    = {植田 暢大 and 波部 英子 and 湯口 彰重 and 河野 誠也 and 川西 康友 and 黒橋 禎夫 and 吉野 幸一郎},
  title     = {実世界における総合的参照解析を目的としたマルチモーダル対話データセットの構築},
  booktitle = {言語処理学会 第29回年次大会},
  year      = {2023},
  address   = {沖縄},
}
```

## References

- [nobu-g/GLIP](https://github.com/nobu-g/GLIP)
- [riken-grp/J-CRe3](https://github.com/riken-grp/J-CRe3)
- [nobu-g/cohesion-analysis](https://github.com/nobu-g/cohesion-analysis)
