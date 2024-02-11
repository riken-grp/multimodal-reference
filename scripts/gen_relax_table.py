import argparse
import io
import subprocess
import sys
from dataclasses import dataclass
from itertools import chain
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
        items = [f"{self.phrase_grounding_model}_{self.model_name}"]
        if self.coref_relax_mode is not None:
            items.append(f"coref_relax_{self.coref_relax_mode}")
        if self.mot_relax_mode is not None:
            items.append(f"mot_relax_{self.mot_relax_mode}")
        return "-".join(items)


@dataclass(frozen=True)
class RatioMetric:
    positive: int
    total: int

    @property
    def ratio(self) -> float:
        if self.total == 0:
            return 0.0
        return self.positive / self.total


def gen_relax_comparison_profile(phrase_grounding_model: str, model_name: str, mot_name: str) -> list[ExpConfig]:
    configs = []
    relax_modes: list[tuple[Optional[str], Optional[str]]] = [
        (None, None),
        ("pred", None),
        ("gold", None),
        (None, f"{mot_name}-max"),
        (None, "gold-max"),
        ("pred", f"{mot_name}-max"),
        ("gold", "gold-max"),
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
    parser.add_argument("glip_name", default="ft2_deberta_b32_new_mixed_1e", help="Config name for GLIP model.")
    parser.add_argument("mot_name", default="detic_th0.3", help="Config name for GLIP model.")
    # https://github.com/gregbanks/python-tabulate#table-format
    parser.add_argument("--table-format", "--table-fmt", default="github", help="Table format.")
    parser.add_argument(
        "--recall-topk", "--topk", type=int, nargs="*", default=[1, 5, 10], help="For calculating Recall@k."
    )
    parser.add_argument(
        "--id-file", type=Path, nargs="+", default=[Path("data/id/test.id")], help="Paths to scenario id file"
    )
    parser.add_argument("--include-precision", action="store_true", help="Include precision in the table.")
    args = parser.parse_args()
    scenario_ids: list[str] = list(chain.from_iterable(path.read_text().splitlines() for path in args.id_file))
    exp_configs = gen_relax_comparison_profile("glip", args.glip_name, args.mot_name)
    data: dict[ExpConfig, list[str]] = {config: [] for config in exp_configs}
    for exp_config in exp_configs:
        exp_name = exp_config.get_name()
        precisions = []
        command = (
            f"{sys.executable} src/evaluation.py"
            " -d data/dataset"
            " -k data/knp"
            " -a data/image_text_annotation"
            f" -p result/mmref/{exp_name}"
            " --prediction-knp-dir result/cohesion"
            f" --scenario-ids {' '.join(scenario_ids)}"
            f" --recall-topk {' '.join(map(str, args.recall_topk))}"
            " --th 0.0"
            " --eval-modes rel"
            " --format csv"
        )
        try:
            output = subprocess.run(command.split(), capture_output=True, check=True, text=True)
        except subprocess.CalledProcessError as e:
            print(e.stderr, file=sys.stderr)
            raise e
        rel_metric_table = pl.read_csv(io.StringIO(output.stdout))
        df_rel = rel_metric_table.filter(pl.col("rel_type") == "=")
        for recall_top_k in args.recall_topk:
            recall = RatioMetric(df_rel[f"recall_pos@{recall_top_k}"][0], df_rel["recall_total"][0])
            data[exp_config].append(f"{recall.ratio:.3f} ({recall.positive})")
        if args.include_precision:
            precisions.append(RatioMetric(df_rel["precision_pos"][0], df_rel["precision_total"][0]))
            data[exp_config].append(f"{precisions[0].ratio:.3f} ({precisions[0].positive} / {precisions[0].total})")

    headers = [f"Recall@{k}" for k in args.recall_topk]
    if args.include_precision:
        headers.append("Precision")
    print(
        tabulate.tabulate(
            [[config.get_name()] + data[config] for config in exp_configs],
            headers=headers,
            tablefmt=args.table_format,
            floatfmt=".3f",
        )
    )


if __name__ == "__main__":
    main()
