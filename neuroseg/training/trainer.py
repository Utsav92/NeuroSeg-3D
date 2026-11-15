"""End-to-end training loop: accumulation, clipping, schedulers, sliding-window validation,
checkpointing, early stopping, resume and crash recovery."""

from __future__ import annotations

import json
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import neuroseg
from neuroseg.data.dataset import FullVolumeDataset, PatchDataset, load_processed
from neuroseg.data.patch_sampler import PatchSampler
from neuroseg.data.pipeline import load_splits, splits_hash
from neuroseg.data.transforms import Augmenter
from neuroseg.evaluation.metrics import batch_region_dice
from neuroseg.inference.sliding_window import sliding_window_predict
from neuroseg.models import build_model
from neuroseg.training.checkpointing import load_checkpoint, save_checkpoint
from neuroseg.training.losses import build_loss, class_weights_from_counts
from neuroseg.training.optimizers import build_optimizer
from neuroseg.training.scheduler import build_scheduler
from neuroseg.utils.device import empty_cache, memory_usage, select_device
from neuroseg.utils.logging import MetricLogger, get_logger, write_json
from neuroseg.utils.seed import seed_everything, worker_init_fn


class Trainer:
    def __init__(self, cfg: dict, resume: bool = False):
        self.cfg = cfg
        self.exp = cfg["experiment"]
        self.tcfg = cfg["train"]
        self.seed = int(cfg.get("seed", 42))
        seed_everything(self.seed, bool(self.tcfg.get("deterministic", False)))
        self.out = Path(cfg.get("output_dir", "outputs")) / "checkpoints" / self.exp
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = get_logger(
            "neuroseg.train", Path(cfg.get("output_dir", "outputs")) / "logs" / f"{self.exp}.log"
        )
        self.device = select_device(self.tcfg.get("device", "auto"))
        self.log.info(f"experiment={self.exp} device={self.device}")

        d = cfg["data"]
        self.modalities = d["modalities"]
        self.patch = tuple(d["patch_size"])
        self.splits = load_splits(d["splits_dir"])
        pdir = Path(d["processed_dir"])
        self.dirs = {k: [pdir / c for c in self.splits[k]["cases"]] for k in ("train", "validation", "test")}
        for k in ("train", "validation"):
            if not self.dirs[k]:
                raise RuntimeError(f"split '{k}' is empty")

        mcfg = dict(cfg["model"])
        mcfg["in_channels"] = len(self.modalities)
        mcfg.setdefault("out_channels", 4)
        if self.tcfg.get("gradient_checkpointing") is not None:
            mcfg["gradient_checkpointing"] = bool(self.tcfg["gradient_checkpointing"])
        self.model_cfg = mcfg
        self.model = build_model(mcfg).to(self.device)
        self.n_params = self.model.count_parameters()
        self.log.info(f"params={self.n_params:,}")

        # class weights from TRAIN split only
        counts = np.zeros(4)
        for dd in self.dirs["train"]:
            counts += np.array(load_processed(dd)["meta"]["label_counts"], float)
        self.class_counts = counts.tolist()
        cw = class_weights_from_counts(counts) if cfg.get("loss", {}).get("ce_class_weights") else None
        spacing = tuple(load_processed(self.dirs["train"][0])["meta"]["spacing_mm"])
        self.spacing = spacing
        self.criterion = build_loss(cfg.get("loss", {}), cw, spacing).to(self.device)

        self.optimizer = build_optimizer(self.model.parameters(), cfg.get("optim", {}))
        self.epochs = int(self.tcfg["epochs"])
        self.sched_cfg = cfg.get("scheduler", {"name": "cosine"})
        self.scheduler = build_scheduler(self.optimizer, self.sched_cfg, self.epochs)
        self.amp = bool(self.tcfg.get("amp", False)) and self.device.type == "cuda"
        if self.tcfg.get("amp") and not self.amp:
            self.log.info("amp requested but only enabled on CUDA; running fp32")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp) if self.amp else None

        self.metric_logger = MetricLogger(self.out, tensorboard=bool(self.tcfg.get("tensorboard", True)))
        self.start_epoch, self.best, self.bad_rounds = 0, -1.0, 0
        self.history: list[dict] = []
        if resume and (self.out / "last.pt").exists():
            self._resume()

    # ------------------------------------------------------------------ data
    def _loader(self) -> DataLoader:
        d = self.cfg["data"]
        aug = Augmenter(d.get("augmentation", "full"), d.get("augmentation_cfg"))
        sampler = PatchSampler(self.patch, d.get("sampler_probs"), d.get("jitter", 0.25))
        ds = PatchDataset(
            self.dirs["train"], self.patch, int(d["samples_per_epoch"]), self.modalities, aug, sampler,
            seed=self.seed, cache_cases=int(d.get("cache_cases", 0)),
        )  # fmt: skip
        self.train_ds = ds
        nw = int(self.tcfg.get("num_workers", 0))
        return DataLoader(
            ds, batch_size=int(self.tcfg.get("batch_size", 1)), shuffle=False, num_workers=nw, drop_last=False,
            worker_init_fn=worker_init_fn if nw else None, pin_memory=self.device.type == "cuda",
        )  # fmt: skip

    # ------------------------------------------------------------------ checkpoints
    def _state(self, epoch: int, extra: dict | None = None) -> dict:
        return {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "best": self.best,
            "bad_rounds": self.bad_rounds,
            "history": self.history,
            "config": json.loads(json.dumps(self.cfg, default=str)),
            "model_cfg": self.model_cfg,
            "modalities": self.modalities,
            "patch_size": list(self.patch),
            "spacing_mm": list(self.spacing),
            "class_names": ["background", "necrotic_nonenhancing", "edema", "enhancing"],
            "class_counts_train": self.class_counts,
            "n_params": self.n_params,
            "splits_hash": splits_hash(self.splits),
            "versions": {
                "neuroseg": neuroseg.__version__,
                "torch": str(torch.__version__),
                "python": platform.python_version(),
            },
            "normalization": self.cfg.get(
                "preprocess", {"method": "per-volume robust z-score on foreground"}
            ),
            "label_info": self.cfg.get("label_info", {}),
            "dataset": self.cfg.get("data", {}).get("dataset"),
            **(extra or {}),
        }

    def _resume(self) -> None:
        ck = load_checkpoint(self.out / "last.pt")
        self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        if self.scheduler and ck.get("scheduler"):
            self.scheduler.load_state_dict(ck["scheduler"])
        self.start_epoch, self.best, self.bad_rounds = ck["epoch"] + 1, ck["best"], ck.get("bad_rounds", 0)
        self.history = ck.get("history", [])
        self.log.info(f"resumed from epoch {ck['epoch']} (best={self.best:.4f})")

    # ------------------------------------------------------------------ loops
    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        accum = int(self.tcfg.get("grad_accum", 1))
        clip = float(self.tcfg.get("grad_clip", 1.0))
        total, n, step = 0.0, 0, 0
        self.optimizer.zero_grad(set_to_none=True)
        nb = len(loader)
        for i, batch in enumerate(loader):
            x = batch["image"].to(self.device, non_blocking=True)
            y = batch["label"].to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                logits = self.model(x)
                loss = self.criterion(logits.float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at batch {i}: {loss.item()}")
            (loss / accum).backward() if self.scaler is None else self.scaler.scale(loss / accum).backward()
            total += float(loss.detach())
            n += 1
            step += 1
            if step % accum == 0 or i + 1 == nb:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
        return total / max(n, 1)

    @torch.no_grad()
    def validate(self, max_cases: int | None = None) -> dict:
        """Whole-volume sliding-window validation on VALIDATION subjects only."""
        self.model.eval()
        ds = FullVolumeDataset(self.dirs["validation"][: max_cases or None], self.modalities)
        overlap = float(self.tcfg.get("val_overlap", 0.5))
        losses: list[float] = []
        dices: dict[str, list[float]] = {"WT": [], "TC": [], "ET": []}
        for i in range(len(ds)):
            s = ds[i]
            probs = sliding_window_predict(
                self.model,
                s["image"],
                self.patch,
                overlap,
                int(self.tcfg.get("sw_batch_size", 1)),
                self.device,
            )
            y = s["label"]
            logits = torch.log(probs.clamp_min(1e-7))[None].to(self.device)
            losses.append(float(self.criterion(logits, y[None].to(self.device))))
            pred = probs.argmax(0).numpy()
            for r, v in batch_region_dice(pred, y.numpy()).items():
                dices[r].append(v)
        res = {"val_loss": float(np.mean(losses))}
        for r in dices:
            res[f"val_dice_{r}"] = float(np.mean(dices[r]))
        res["val_mean_dice"] = float(np.mean([res[f"val_dice_{r}"] for r in dices]))
        res["val_cases"] = len(ds)
        return res

    def fit(self) -> dict:
        loader = self._loader()
        patience = int(self.tcfg.get("early_stopping", {}).get("patience", 10))
        min_delta = float(self.tcfg.get("early_stopping", {}).get("min_delta", 1e-4))
        early = bool(self.tcfg.get("early_stopping", {}).get("enabled", True))
        val_every = int(self.tcfg.get("val_interval", 1))
        stop_reason = "max_epochs"
        t_start = time.perf_counter()
        epoch = self.start_epoch - 1
        try:
            for epoch in range(self.start_epoch, self.epochs):
                t0 = time.perf_counter()
                self.train_ds.set_epoch(epoch)
                tr_loss = self._train_epoch(loader)
                row = {
                    "train_loss": tr_loss,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "n_params": self.n_params,
                }
                do_val = (epoch + 1) % val_every == 0 or epoch + 1 == self.epochs
                if do_val:
                    row.update(self.validate(self.tcfg.get("val_max_cases")))
                    metric = row["val_mean_dice"]
                    if metric > self.best + min_delta:
                        self.best, self.bad_rounds = metric, 0
                        save_checkpoint(self.out / "best.pt", self._state(epoch, {"best_epoch": epoch}))
                        row["is_best"] = 1
                    else:
                        self.bad_rounds += 1
                    if self.scheduler is not None and self.sched_cfg.get("name") == "plateau":
                        self.scheduler.step(metric)
                if self.scheduler is not None and self.sched_cfg.get("name") == "cosine":
                    self.scheduler.step()
                row["epoch_seconds"] = time.perf_counter() - t0
                row.update(memory_usage(self.device))
                self.history.append({"epoch": epoch, **row})
                self.metric_logger.log(epoch, row)
                save_checkpoint(self.out / "last.pt", self._state(epoch))
                self.log.info(
                    " ".join(
                        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in {"epoch": epoch, **row}.items()
                        if k
                        in (
                            "epoch",
                            "train_loss",
                            "val_loss",
                            "val_dice_WT",
                            "val_dice_TC",
                            "val_dice_ET",
                            "val_mean_dice",
                            "epoch_seconds",
                            "rss_gb",
                        )
                    )
                )
                empty_cache(self.device)
                if early and do_val and self.bad_rounds >= patience:
                    stop_reason = f"early_stopping(patience={patience})"
                    self.log.info(f"early stopping at epoch {epoch}")
                    break
        except BaseException as e:  # crash recovery: persist state, then re-raise
            save_checkpoint(self.out / "crash.pt", self._state(epoch, {"crash": f"{type(e).__name__}: {e}"}))
            self.log.error(
                f"training interrupted ({type(e).__name__}); emergency checkpoint saved. resume with --resume\n{traceback.format_exc()}"
            )
            raise
        finally:
            self.metric_logger.close()
        record = {
            "experiment": self.exp,
            "seed": self.seed,
            "device": str(self.device),
            "stop_reason": stop_reason,
            "epochs_run": epoch + 1,
            "best_val_mean_dice": self.best if self.best >= 0 else None,
            "selection_metric": "val_mean_dice (validation split, whole-volume sliding window)",
            "checkpoint_best": str(self.out / "best.pt") if (self.out / "best.pt").exists() else None,
            "checkpoint_last": str(self.out / "last.pt"),
            "n_params": self.n_params,
            "wall_seconds": time.perf_counter() - t_start,
            "splits_hash": splits_hash(self.splits),
            "config": json.loads(json.dumps(self.cfg, default=str)),
            "test_set_used": False,
        }
        write_json(self.out / "run_record.json", record)
        return record
