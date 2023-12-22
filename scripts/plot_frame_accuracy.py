import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Hashable
from pathlib import Path

import plotly.express as px
import polars as pl
from rhoknp import Document

RECALL_TOP_KS = [*range(1, 11), 100]


class IdMapper:
    """Consistently map ids of any type to integers."""

    def __init__(self):
        self._id_to_int: dict[Hashable, int] = {}
        self._int_to_id: dict[int, Hashable] = {}
        self._next_int: int = 0

    def __len__(self) -> int:
        return len(self._id_to_int)

    def __contains__(self, id_: object) -> bool:
        return id_ in self._id_to_int

    def __getitem__(self, id_: object) -> int:
        if id_ not in self._id_to_int:
            self._id_to_int[id_] = self._next_int
            self._int_to_id[self._next_int] = id_
            self._next_int += 1
        return self._id_to_int[id_]

    def map(self, id_: object) -> int:
        return self[id_]


def main() -> None:
    project_root = Path.cwd()
    interpreter = sys.executable
    scenario_ids: list[str] = (
        project_root.joinpath("data/id/test.id")
        .read_text()
        .splitlines()
        # + project_root.joinpath("data/id/valid.id").read_text().splitlines()
    )
    exp_name = sys.argv[1]
    with tempfile.TemporaryDirectory() as out_dir:
        subprocess.run(
            (
                f"{interpreter} src/evaluation.py -p result/mmref/{exp_name} --scenario-ids {' '.join(scenario_ids)}"
                f" --recall-topk {' '.join(map(str, RECALL_TOP_KS))} --th 0.0 --raw-result-csv {out_dir}/raw_result.csv"
            ).split(),
            cwd=project_root,
            check=True,
        )
        df_raw_result = pl.read_csv(f"{out_dir}/raw_result.csv")
    output_dir = Path("data") / "frame_accuracy"
    output_dir.mkdir(exist_ok=True)
    for scenario_id in scenario_ids:
        visualize(df_raw_result, scenario_id, output_dir / f"{scenario_id}.pdf")


def visualize(comparison_table: pl.DataFrame, scenario_id: str, output_file: Path) -> None:
    comparison_table = (
        comparison_table.filter(pl.col("rel_type") == "=")
        .filter(pl.col("class_name") != "")
        .filter(pl.col("scenario_id") == scenario_id)
        .drop(["scenario_id", "rel_type"])
    )
    gold_document = Document.from_knp(Path(f"data/knp/{scenario_id}.knp").read_text())
    sid2sentence = {sentence.sid: sentence for sentence in gold_document.sentences}
    id_mapper = IdMapper()
    plot_table = []
    instance_id_to_label = {}
    instance_id_to_texts = defaultdict(set)
    for row in comparison_table.to_dicts():
        rank = -1
        for recall_top_k in RECALL_TOP_KS:
            metric_suffix = f"@{recall_top_k}"
            if row[f"recall_pos{metric_suffix}"] > 0:
                rank = recall_top_k
                break
        base_phrase = sid2sentence[row["sid"]].base_phrases[row["base_phrase_index"]]
        instance_id: str = str(id_mapper.map(row["instance_id_or_pred_idx"]))
        instance_id_to_label[instance_id] = f'{row["class_name"]}_{instance_id}'
        instance_id_to_texts[instance_id].add((base_phrase.text, base_phrase.global_index))
        plot_table.append(
            {
                "image_id": int(row["image_id"]),
                "instance_id": instance_id,
                "rank": rank,
            }
        )
    fig = px.line(
        pl.DataFrame(plot_table),
        x="image_id",
        y="rank",
        color="instance_id",
        labels={"x": "Frame Number", "y": "Rank"},
        markers=True,
        width=960,
        height=480,
    )

    for instance_id, texts in instance_id_to_texts.items():
        sorted_texts: list[str] = [x[0] for x in sorted(texts, key=lambda x: x[1])]
        instance_id_to_label[instance_id] += f" ({', '.join(sorted_texts)})"

    fig.for_each_trace(
        lambda t: t.update(
            name=instance_id_to_label[t.name],
            legendgroup=instance_id_to_label[t.name],
            hovertemplate=t.hovertemplate.replace(t.name, instance_id_to_label[t.name]),
        )
    )

    fig.update_layout(
        barmode="overlay",
        title=f"{scenario_id}",
        # https://plotly.com/python/reference/layout/
        xaxis=dict(
            title="Frame",
            dtick=10,
        ),
        yaxis=dict(
            type="category",
            title="Rank",
            categoryarray=list(reversed([*range(1, 11), 100, -1])),
            tickmode="array",
            tickvals=list(reversed([*range(1, 11), 100, -1])),
            ticktext=list(reversed([*range(-1, 11), "≦100", "Missed"])),
        ),
        margin=dict(l=5, r=5, b=5, t=50, pad=0),
    )

    # https://github.com/plotly/plotly.py/issues/3469
    import plotly.io as pio

    pio.kaleido.scope.mathjax = None
    fig.write_image(output_file)


if __name__ == "__main__":
    main()
