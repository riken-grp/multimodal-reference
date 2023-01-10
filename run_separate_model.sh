#!/usr/bin/env bash

set -euo pipefail

exp_name=mdetr_mixed_1e_flickr_1e
scenario_ids=$(ls data/dataset)

for scenario_id in $scenario_ids; do
  echo "Running scenario $scenario_id"
  poetry run python src/prediction_writer.py -cn server dataset_dir="data/dataset/${scenario_id}" gold_knp_file="data/knp/${scenario_id}.knp" prediction_dir="result/${exp_name}"
done
