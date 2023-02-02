#!/usr/bin/env bash

set -euo pipefail

exp_name=mdetr_mixed_2e_flickr_2e
scenario_ids=$(cat full_annotated.txt)

for scenario_id in $scenario_ids; do
  echo "Running scenario $scenario_id"
  if [[ -f "result/${exp_name}/${scenario_id}.json" ]]; then
    echo "Skip"
    continue
  fi
  poetry run python src/prediction_writer.py -cn server dataset_dir="data/dataset/${scenario_id}" gold_knp_file="data/knp/${scenario_id}.knp" prediction_dir="result/${exp_name}"
done
