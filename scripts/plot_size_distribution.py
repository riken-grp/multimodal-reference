import argparse
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean

import plotly.express as px
import polars as pl

RECALL_TOP_KS = [*range(1, 11), 100]
IMAGE_SIZE = 1920 * 1080


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, help="Experiment name (directory name under result/mmref)")
    parser.add_argument(
        "--id-file", type=Path, nargs="+", default=[Path("data/id/test.id")], help="Paths to scenario id file"
    )
    args = parser.parse_args()
    scenario_ids: list[str] = sum((path.read_text().splitlines() for path in args.id_file), [])
    with tempfile.TemporaryDirectory() as out_dir:
        subprocess.run(
            (
                f"{sys.executable} src/evaluation.py -p result/mmref/{args.exp_name} --scenario-ids {' '.join(scenario_ids)}"
                f" --recall-topk {' '.join(map(str, RECALL_TOP_KS))} --th 0.0 --raw-result-csv {out_dir}/raw_result.csv"
            ).split(),
            cwd=Path.cwd(),
            check=True,
        )
        df_raw_result = pl.read_csv(f"{out_dir}/raw_result.csv")
    output_dir = Path("data") / "size_distribution"
    output_dir.mkdir(exist_ok=True)
    visualize(df_raw_result, output_dir / f"{args.exp_name}.pdf")


def visualize(comparison_table: pl.DataFrame, output_file: Path) -> None:
    comparison_table = (
        comparison_table.filter(pl.col("rel_type") == "=")
        .filter(pl.col("class_name") != "")
        .drop(["scenario_id", "image_id", "sid", "base_phrase_index", "rel_type", "instance_id_or_pred_idx"])
    )
    ranks = []
    sizes = []
    for row in comparison_table.to_dicts():
        rank = -1
        for recall_topk in RECALL_TOP_KS:
            metric_suffix = f"@{recall_topk}"
            if row[f"recall_pos{metric_suffix}"] > 0:
                rank = recall_topk
                break
        ranks.append(rank)
        sizes.append(row["width"] * row["height"] / IMAGE_SIZE)

    # Calculate averages
    average_sizes_per_rank = defaultdict(list)
    for rank, size in zip(ranks, sizes):
        average_sizes_per_rank[rank].append(size)

    averages = {rank: mean(sizes) for rank, sizes in average_sizes_per_rank.items()}

    fig = px.scatter(x=sizes, y=ranks, labels={"x": "Frame Occupancy", "y": "Rank"}, log_x=True)

    fig.add_scatter(
        x=list(averages.values()), y=list(averages.keys()), mode="markers", marker=dict(color="red"), name="Average"
    )
    fig.update_layout(
        barmode="overlay",
        # https://plotly.com/python/reference/layout/yaxis/
        yaxis=dict(
            type="category",
            # title="Rank",
            categoryarray=list(reversed([*range(1, 11), 100, -1])),
            tickmode="array",
            tickvals=list(reversed([*range(1, 11), 100, -1])),
            ticktext=list(reversed([*range(-1, 11), "≦100", "Missed"])),
        ),
    )

    # https://github.com/plotly/plotly.py/issues/3469
    import plotly.io as pio

    pio.kaleido.scope.mathjax = None
    fig.write_image(output_file)
    # fig.show()


if __name__ == "__main__":
    main()
