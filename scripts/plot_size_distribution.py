import argparse
import subprocess
import sys
import tempfile
from itertools import chain
from pathlib import Path

import plotly.express as px
import polars as pl

RECALL_TOP_KS = [*range(1, 11), 100]
IMAGE_SIZE = 1920 * 1080


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_names", type=str, nargs="+", help="Experiment name (directory name under result/mmref)")
    parser.add_argument(
        "--id-file", type=Path, nargs="+", default=[Path("data/id/test.id")], help="Paths to scenario id file"
    )
    args = parser.parse_args()
    scenario_ids: list[str] = list(chain.from_iterable(path.read_text().splitlines() for path in args.id_file))
    for exp_name in args.exp_names:
        with tempfile.TemporaryDirectory() as out_dir:
            command = (
                f"{sys.executable} src/evaluation.py -p result/mmref/{exp_name} --scenario-ids {' '.join(scenario_ids)}"
                f" --recall-topk {' '.join(map(str, RECALL_TOP_KS))} --th 0.0 --raw-result-csv {out_dir}/raw_result.csv"
            )
            subprocess.run(command.split(), cwd=Path.cwd(), check=True)
            df_raw_result = pl.read_csv(f"{out_dir}/raw_result.csv")
        output_dir = Path("data") / "size_distribution"
        output_dir.mkdir(exist_ok=True)
        visualize(df_raw_result, output_dir / f"{exp_name}.pdf")


def visualize(comparison_table: pl.DataFrame, output_file: Path) -> None:
    comparison_table = (
        comparison_table.filter(pl.col("rel_type") == "=")
        .filter(pl.col("class_name") != "")
        .drop(["scenario_id", "image_id", "sid", "base_phrase_index", "rel_type", "instance_id_or_pred_idx"])
    )
    data = []
    for row in comparison_table.to_dicts():
        rank = -1
        for recall_topk in RECALL_TOP_KS:
            metric_suffix = f"@{recall_topk}"
            if row[f"recall_pos{metric_suffix}"] > 0:
                rank = recall_topk
                break
        data.append({"rank": rank, "size": row["width"] * row["height"] / IMAGE_SIZE})
    df_size = pl.DataFrame(data)

    fig = px.scatter(df_size, x="size", y="rank", log_x=True)

    # Plot averages
    df_size_average = df_size.group_by("rank").agg(pl.mean("size")).sort(by="rank")
    fig.add_scatter(
        x=df_size_average["size"], y=df_size_average["rank"], mode="markers", marker=dict(color="red"), name="Average"
    )
    fig.update_layout(
        barmode="overlay",
        xaxis=dict(
            title="Frame Occupancy",
        ),
        # https://plotly.com/python/reference/layout/yaxis/
        yaxis=dict(
            type="category",
            title="Rank",
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
