import warnings
from pathlib import Path

import hydra
import luigi
from omegaconf import DictConfig

from tasks.mmref import MultimodalReference

warnings.filterwarnings(
    "ignore",
    message=r'Parameter "(cfg|document)" with value .+ is not of type string\.',
    category=UserWarning,
)


@hydra.main(version_base=None, config_path="../configs")
def main(cfg: DictConfig) -> None:
    if len(cfg.scenario_ids) == 0 and cfg.id_file is not None:
        cfg.scenario_ids = Path(cfg.id_file).read_text().strip().split()
    tasks: list[luigi.Task] = []
    for scenario_id in cfg.scenario_ids:
        tasks.append(MultimodalReference(cfg=cfg, scenario_id=scenario_id))
    luigi.build(tasks, **cfg.luigi)


if __name__ == "__main__":
    main()
