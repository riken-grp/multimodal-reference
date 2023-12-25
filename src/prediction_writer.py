import warnings
from pathlib import Path

import hydra
import luigi
from omegaconf import DictConfig, OmegaConf

from tasks.mmref import MultimodalReference

warnings.filterwarnings(
    "ignore",
    message=r'Parameter "(cfg|document)" with value .+ is not of type string\.',
    category=UserWarning,
)

OmegaConf.register_new_resolver(
    "get", lambda d, k, v: getattr(d, k, v) if k is not None else v, replace=True, use_cache=True
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
