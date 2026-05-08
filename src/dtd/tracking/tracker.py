from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


class Tracker:
    def log(self, metrics: dict[str, float], step: int):
        return

    def log_artifact(self, path: str | Path, name: Optional[str] = None):
        return

    def finish(self):
        return


@dataclass
class CompositeTracker(Tracker):
    trackers: list[Tracker]

    def log(self, metrics: dict[str, float], step: int):
        for t in self.trackers:
            t.log(metrics, step)

    def log_artifact(self, path: str | Path, name: Optional[str] = None):
        for t in self.trackers:
            t.log_artifact(path, name=name)

    def finish(self):
        for t in self.trackers:
            t.finish()


class WandbTracker(Tracker):
    def __init__(self, *, out_dir: Path, project: str, entity: Optional[str], name: str, tags: list[str]):
        try:
            import wandb
        except Exception as e:
            raise RuntimeError("wandb is not installed. `pip install wandb`") from e
        self.wandb = wandb
        self.run = wandb.init(project=project, entity=entity, name=name, dir=str(out_dir), tags=tags)

    def log(self, metrics: dict[str, float], step: int):
        self.wandb.log(metrics, step=step)

    def log_artifact(self, path: str | Path, name: Optional[str] = None):
        p = Path(path)
        art = self.wandb.Artifact(name or p.name, type="artifact")
        art.add_file(str(p))
        self.run.log_artifact(art)

    def finish(self):
        try:
            self.run.finish()
        except Exception:
            pass


class MlflowTracker(Tracker):
    def __init__(self, *, out_dir: Path, experiment: str, tracking_uri: Optional[str], run_name: str):
        try:
            import mlflow
        except Exception as e:
            raise RuntimeError("mlflow is not installed. `pip install mlflow`") from e
        self.mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self.run = mlflow.start_run(run_name=run_name)
        self.out_dir = out_dir

    def log(self, metrics: dict[str, float], step: int):
        self.mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path, name: Optional[str] = None):
        self.mlflow.log_artifact(str(path), artifact_path=name)

    def finish(self):
        try:
            self.mlflow.end_run()
        except Exception:
            pass


def build_tracker(cfg: Any, *, out_dir: str | Path, run_name: str) -> Tracker:
    out_dir = Path(out_dir)
    trackers: list[Tracker] = []

    wcfg = getattr(cfg, "wandb", None)
    if wcfg and getattr(wcfg, "enabled", False):
        trackers.append(
            WandbTracker(
                out_dir=out_dir,
                project=str(getattr(wcfg, "project", "dtd")),
                entity=getattr(wcfg, "entity", None),
                name=run_name,
                tags=list(getattr(wcfg, "tags", []) or []),
            )
        )

    mcfg = getattr(cfg, "mlflow", None)
    if mcfg and getattr(mcfg, "enabled", False):
        trackers.append(
            MlflowTracker(
                out_dir=out_dir,
                experiment=str(getattr(mcfg, "experiment", "dtd")),
                tracking_uri=getattr(mcfg, "tracking_uri", None),
                run_name=run_name,
            )
        )

    return CompositeTracker(trackers) if trackers else Tracker()

