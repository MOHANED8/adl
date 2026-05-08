from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from dtd.training.logging import configure_logging, get_logger


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    configure_logging(str(getattr(cfg, "log_level", "INFO")))
    logger = get_logger(__name__)
    from dtd.analysis import (
        ActivationMonitor,
        ActivationMonitorConfig,
        CollapseMonitorConfig,
        GradientMonitor,
        GradientMonitorConfig,
        LayerCollapseMonitor,
        NeuronSpecializationMonitor,
        RepresentationMonitor,
        RepresentationMonitorConfig,
        SelfSupervisedConsistencyMonitor,
        SelfSupervisedMonitorConfig,
        SpecializationMonitorConfig,
        UncertaintyMonitor,
        UncertaintyMonitorConfig,
    )
    from dtd.datasets import DatasetSpec, build_dataloaders
    from dtd.hooks import HookSpec
    from dtd.models import ModelSpec, build_model
    from dtd.monitoring import EarlyWarningSystem
    from dtd.tracking import build_tracker
    from dtd.training import OptimConfig, RuntimeConfig, TrainConfig, Trainer
    from dtd.training.distributed import cleanup_runtime, setup_runtime

    logger.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg))

    runtime_cfg = RuntimeConfig(
        device=str(cfg.training.runtime.device),
        multi_gpu=bool(getattr(cfg.training.runtime, "multi_gpu", False)),
        distributed=bool(getattr(cfg.training.runtime, "distributed", False)),
        backend=str(getattr(cfg.training.runtime, "backend", "nccl")),
        amp=bool(getattr(cfg.training.runtime, "amp", False)),
        amp_dtype=str(getattr(cfg.training.runtime, "amp_dtype", "float16")),
        grad_clip_norm=(
            float(cfg.training.runtime.grad_clip_norm)
            if getattr(cfg.training.runtime, "grad_clip_norm", None) is not None
            else None
        ),
        seed=int(getattr(cfg.training.runtime, "seed", 42)),
    )
    dist_ctx = setup_runtime(runtime_cfg)

    try:
        ds = cfg.dataset
        loaders = build_dataloaders(
            train=DatasetSpec(
                name=str(ds.name),
                root=str(ds.root),
                split=str(ds.train.split),
                batch_size=int(ds.train.batch_size),
                num_workers=int(ds.train.num_workers),
                image_size=int(ds.image_size),
                aug=str(ds.aug),
                cache_dir=str(ds.cache_dir),
                label_noise_p=float(getattr(ds, "label_noise_p", 0.0)),
                label_noise_seed=int(getattr(ds, "label_noise_seed", 0)),
            ),
            val=DatasetSpec(
                name=str(ds.name),
                root=str(ds.root),
                split=str(ds.val.split),
                batch_size=int(ds.val.batch_size),
                num_workers=int(ds.val.num_workers),
                image_size=int(ds.image_size),
                aug=str(ds.aug),
                cache_dir=str(ds.cache_dir),
                shuffle=False,
            ),
        )

        ms = cfg.model
        hs = ms.hooks
        hook_spec = HookSpec(
            module_globs=tuple(hs.module_globs),
            capture_inputs=bool(hs.capture_inputs),
            capture_outputs=bool(hs.capture_outputs),
            capture_output_grads=bool(hs.capture_output_grads),
            capture_attention=bool(hs.capture_attention),
            detach=bool(hs.detach),
            cpu=bool(hs.cpu),
            max_per_module=int(hs.max_per_module),
        )
        model = build_model(
            ModelSpec(
                name=str(ms.name),
                num_classes=int(ds.num_classes),
                pretrained=bool(ms.pretrained),
                hook_spec=hook_spec,
            )
        )

        diag = cfg.diagnostics
        monitors = []
        if diag.activation.enabled:
            monitors.append(ActivationMonitor(ActivationMonitorConfig(feature_mode=str(diag.activation.feature_mode))))
        if diag.gradient.enabled:
            monitors.append(GradientMonitor(GradientMonitorConfig()))
        if diag.representation.enabled:
            monitors.append(
                RepresentationMonitor(RepresentationMonitorConfig(module_name=str(diag.representation.module_name)))
            )
        if getattr(diag, "uncertainty", None) and diag.uncertainty.enabled:
            monitors.append(
                UncertaintyMonitor(
                    UncertaintyMonitorConfig(
                        sample_batches=int(diag.uncertainty.sample_batches),
                        bins=int(diag.uncertainty.bins),
                        fit_temperature_scaling=bool(diag.uncertainty.fit_temperature_scaling),
                        temperature_steps=int(diag.uncertainty.temperature_steps),
                        temperature_lr=float(diag.uncertainty.temperature_lr),
                    )
                )
            )
        if getattr(diag, "specialization", None) and diag.specialization.enabled:
            monitors.append(
                NeuronSpecializationMonitor(
                    SpecializationMonitorConfig(
                        module_name=str(diag.specialization.module_name),
                        feature_mode=str(diag.specialization.feature_mode),
                        topk_fraction=float(diag.specialization.topk_fraction),
                    )
                )
            )
        if getattr(diag, "collapse", None) and diag.collapse.enabled:
            monitors.append(
                LayerCollapseMonitor(
                    CollapseMonitorConfig(
                        module_name=str(diag.collapse.module_name),
                        feature_mode=str(diag.collapse.feature_mode),
                        collapse_rank_threshold=float(diag.collapse.collapse_rank_threshold),
                    )
                )
            )
        if getattr(diag, "self_supervised", None) and diag.self_supervised.enabled:
            monitors.append(
                SelfSupervisedConsistencyMonitor(
                    SelfSupervisedMonitorConfig(
                        module_name=str(diag.self_supervised.module_name),
                        noise_std=float(diag.self_supervised.noise_std),
                    )
                )
            )
        if diag.early_warning.enabled:
            monitors.append(EarlyWarningSystem(mode=str(diag.early_warning.mode)))

        out_dir = str(cfg.exp.out_dir)
        tcfg = cfg.training
        trainer_cfg = TrainConfig(
            out_dir=out_dir,
            epochs=int(tcfg.epochs),
            optim=OptimConfig(
                name=str(tcfg.optim.name),
                lr=float(tcfg.optim.lr),
                weight_decay=float(tcfg.optim.weight_decay),
                momentum=float(getattr(tcfg.optim, "momentum", 0.9)),
            ),
            log_every_steps=int(tcfg.log_every_steps),
            monitor_every_steps=int(tcfg.monitor_every_steps),
            runtime=runtime_cfg,
        )

        tracker = build_tracker(cfg.tracking, out_dir=out_dir, run_name=str(cfg.exp.name))

        trainer = Trainer(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders.get("val"),
            cfg=trainer_cfg,
            dist_ctx=dist_ctx,
            monitors=monitors,
            tracker=tracker,
        )
        trainer.fit()
    except Exception:
        logger.exception("Training failed.")
        raise
    finally:
        cleanup_runtime(dist_ctx)


if __name__ == "__main__":
    main()

