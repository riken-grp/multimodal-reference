import argparse
import io
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import polars as pl
import tabulate


@dataclass(frozen=True)
class ExpConfig:
    phrase_grounding_model: str  # mdetr or glip
    model_name: str
    coref_relax_mode: Optional[str]  # None, "pred", or "gold"
    mot_relax_mode: Optional[str]  # None, "pred", or "gold"

    def get_name(self) -> str:
        return "-".join(
            [
                f"{self.phrase_grounding_model}_{self.model_name}",
                f"coref_{self.coref_relax_mode}",
                f"mot_{self.mot_relax_mode}",
            ]
        )


@dataclass(frozen=True)
class EvaluationConfig:
    id_file: Path
    # recall_topk: int


@dataclass(frozen=True)
class RatioMetric:
    positive: int
    total: int

    @property
    def ratio(self) -> float:
        if self.total == 0:
            return 0.0
        return self.positive / self.total


def gen_relax_comparison_profile(phrase_grounding_model: str, model_name: str) -> list[ExpConfig]:
    configs = []
    relax_modes: list[tuple[Optional[str], Optional[str]]] = [
        (None, None),
        ("pred", None),
        ("gold", None),
        (None, "pred"),
        (None, "gold"),
        ("pred", "pred"),
        ("gold", "gold"),
    ]
    for coref_relax_mode, mot_relax_mode in relax_modes:
        configs.append(
            ExpConfig(
                phrase_grounding_model=phrase_grounding_model,
                model_name=model_name,
                coref_relax_mode=coref_relax_mode,
                mot_relax_mode=mot_relax_mode,
            )
        )
    return configs


def main():
    parser = argparse.ArgumentParser()
    # https://github.com/gregbanks/python-tabulate#table-format
    parser.add_argument("--table-format", default="github", help="Table format.")
    # parser.add_argument("--recall-topk", "--topk", type=int, default=-1, help="For calculating Recall@k.")
    args = parser.parse_args()
    eval_config = EvaluationConfig(
        id_file=Path("data/id/test.id"),
    )
    scenario_ids: list[str] = eval_config.id_file.read_text().splitlines()
    exp_configs = gen_relax_comparison_profile("glip", "ft1")
    data: dict[ExpConfig, list[str]] = {config: [] for config in exp_configs}
    for exp_config in exp_configs:
        exp_name = exp_config.get_name()
        precisions = []
        for recall_topk in (1, 5, 10):
            output = subprocess.run(
                (
                    f"{sys.executable} src/evaluation.py"
                    f" -d data/dataset"
                    f" -k data/knp"
                    f" -a data/image_text_annotation"
                    f" -p result/mmref/{exp_name}"
                    f" --prediction-knp-dir result/cohesion"
                    f" --scenario-ids {' '.join(scenario_ids)}"
                    f" --recall-topk {recall_topk}"
                    f" --eval-modes rel"
                    f" --format csv"
                ).split(),
                capture_output=True,
                check=True,
                text=True,
            )
            rel_metric_table = pl.read_csv(io.StringIO(output.stdout))
            df_rel = rel_metric_table.filter(pl.col("relation_type") == "=")
            recall = RatioMetric(df_rel["recall_pos"][0], df_rel["recall_total"][0])
            precisions.append(RatioMetric(df_rel["precision_pos"][0], df_rel["precision_total"][0]))
            data[exp_config].append(f"{recall.ratio:.3f} ({recall.positive})")
        data[exp_config].append(f"{precisions[0].ratio:.3f} ({precisions[0].positive} / {precisions[0].total})")

    print(
        tabulate.tabulate(
            [[config.get_name()] + data[config] for config in exp_configs],
            headers=["Recall@1", "Recall@5", "Recall@10", "Precision"],
            tablefmt=args.table_format,
            floatfmt=".3f",
        )
    )


if __name__ == "__main__":
    main()
