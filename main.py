"""
Traffic Sign Recognition — Monolithic CLI Entry Point.

Usage examples:
    # Train on DOT dataset with ResNet10
    python main.py train --model resnet10 --stage full --verbose

    # Train baseline on geometric stage only
    python main.py train --model baseline --stage geometric

    # Train with early stopping and custom epochs
    python main.py train --model advanced --epochs 30 --patience 10

    # Evaluate a saved checkpoint
    python main.py evaluate --model resnet10 --split test --checkpoint checkpoints/best_model.pth

    # Show device info
    python main.py info
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Enable CPU fallback for MPS ops not yet implemented (e.g., grid_sampler_2d_backward)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.device import get_device, device_info
from utils.config import Config, DataConfig, DATA_DIR, CHECKPOINT_DIR, LOG_DIR


def build_model(name: str, num_classes: int) -> nn.Module:
    """Instantiate a model by name."""
    if name == "baseline":
        from models.baseline import BaselineCNN
        return BaselineCNN(num_classes=num_classes)
    elif name == "advanced":
        from models.advanced import AdvancedCNN
        return AdvancedCNN(num_classes=num_classes)
    elif name == "resnet10":
        from models.resnet import ResNet10
        return ResNet10(num_classes=num_classes)
    elif name == "orion":
        from models.orion_vlm import OrionVLMStub
        return OrionVLMStub(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")


def build_dot_loaders(batch_size: int = 8, num_workers: int = 0):
    """
    Build DataLoaders for supervised training + optional domain adaptation.

    Supervised datasets (DOT, GTSRB) contribute to the CE loss.
    LISA and BDD100K provide ``DOMAIN_LABEL=-1`` crops for domain
    adversarial training; they are returned under the ``"domain"`` key
    and are excluded from the classification loss.

    Returns dict with keys: ``train``, ``val``, ``test``, ``ood``,
    and optionally ``domain`` (if LISA / BDD100K data found).
    """
    import zipfile as _zipfile
    from data.datasets import (
        DOTDataset, GTSRBDataset, LISADataset, BDD100KDataset,
        UnifiedTrafficSignDataset, DOMAIN_LABEL,
    )
    from data.transforms import gtsrb_train_transform, bdd100k_train_transform, eval_transform
    from data.dataloaders import stratified_split

    supervised_train: list = []
    supervised_eval:  list = []
    domain_datasets:  list = []

    # ── DOT (canonical reference, always loaded) ──
    dot_root = DATA_DIR / "DOT"
    if dot_root.exists():
        dot_train = DOTDataset(dot_root, transform=gtsrb_train_transform())
        dot_eval  = DOTDataset(dot_root, transform=eval_transform())
        if len(dot_eval) > 0:
            supervised_train.append(dot_train)
            supervised_eval.append(dot_eval)
            print(f"  DOT:     {len(dot_eval):6d} images")
    else:
        print(f"  WARNING: DOT not found at {dot_root}")

    # ── GTSRB (Kaggle CSV format) ──
    gtsrb_root = DATA_DIR / "gtsrb"
    if not gtsrb_root.exists():
        gtsrb_root = DATA_DIR / "gtsrb-german-traffic-sign"  # legacy fallback
    if gtsrb_root.exists():
        gtsrb_train = GTSRBDataset(
            gtsrb_root, transform=gtsrb_train_transform(),
            split="train", download=False, dot_only=True,
        )
        gtsrb_eval = GTSRBDataset(
            gtsrb_root, transform=eval_transform(),
            split="test", download=False, dot_only=True,
        )
        if len(gtsrb_train) > 0:
            supervised_train.append(gtsrb_train)
        if len(gtsrb_eval) > 0:
            supervised_eval.append(gtsrb_eval)
        print(f"  GTSRB:   {len(gtsrb_train):6d} train, {len(gtsrb_eval):6d} eval (DOT-mapped)")
    else:
        print(f"  WARNING: GTSRB not found at {gtsrb_root}")

    # ── LISA (extract from zip on first use; domain adaptation only) ──
    lisa_base = DATA_DIR / "lisa"
    lisa_extracted = lisa_base / "extracted"
    lisa_zip = lisa_base / "lisa-traffic-light-dataset.zip"
    if not lisa_extracted.exists() and lisa_zip.exists():
        print("  Extracting LISA zip (one-time) ...")
        lisa_extracted.mkdir(parents=True, exist_ok=True)
        with _zipfile.ZipFile(lisa_zip) as z:
            z.extractall(lisa_extracted)
        print(f"  LISA extracted to {lisa_extracted}")
    lisa_dir = (
        lisa_extracted if lisa_extracted.exists()
        else lisa_base if (lisa_base / "Annotations").exists()
        else None
    )
    if lisa_dir is not None:
        lisa_domain = LISADataset(
            lisa_dir,
            transform=bdd100k_train_transform(),
            domain_only=True,
        )
        if len(lisa_domain) > 0:
            domain_datasets.append(lisa_domain)
            print(f"  LISA:    {len(lisa_domain):6d} crops (domain-only)")

    # ── BDD100K (on-the-fly crop extraction; domain adaptation only) ──
    bdd100k_root = DATA_DIR / "BDD_100K"
    if bdd100k_root.exists():
        print("  Scanning BDD100K annotations (may take a few seconds) ...")
        bdd_domain = BDD100KDataset(
            bdd100k_root,
            transform=bdd100k_train_transform(),
            split="train",
            domain_only=True,
            min_crop_size=16,
        )
        if len(bdd_domain) > 0:
            domain_datasets.append(bdd_domain)
            print(f"  BDD100K: {len(bdd_domain):6d} sign crops (domain-only)")
        else:
            print("  BDD100K: no crops found (missing 100k/ directory or annotations)")

    if not supervised_train:
        print("ERROR: No supervised datasets found. Check DATA_DIR in .env")
        sys.exit(1)

    combined_train = (
        supervised_train[0] if len(supervised_train) == 1
        else UnifiedTrafficSignDataset(supervised_train)
    )
    combined_eval = (
        supervised_eval[0] if len(supervised_eval) == 1
        else UnifiedTrafficSignDataset(supervised_eval)
    )

    print(f"\n  Supervised: {len(combined_train)} train samples, {len(combined_eval)} eval samples")

    # Split eval into val / test / ood (40% / 30% / 30%)
    splits = stratified_split(combined_eval, ratios=(0.0, 0.4, 0.3, 0.3))
    _, val_sub, test_sub, ood_sub = splits

    loader_kw = dict(batch_size=batch_size, num_workers=num_workers)
    loaders: dict = {
        "train": DataLoader(combined_train, shuffle=True,  drop_last=False, **loader_kw),
        "val":   DataLoader(val_sub,        shuffle=False, **loader_kw),
        "test":  DataLoader(test_sub,       shuffle=False, **loader_kw),
        "ood":   DataLoader(ood_sub,        shuffle=False, **loader_kw),
    }

    if domain_datasets:
        combined_domain = (
            domain_datasets[0] if len(domain_datasets) == 1
            else UnifiedTrafficSignDataset(domain_datasets)
        )
        loaders["domain"] = DataLoader(
            combined_domain, shuffle=True, **loader_kw,
        )
        print(f"  Domain:   {len(combined_domain)} total crops (LISA + BDD100K)")

    print()
    for name, loader in loaders.items():
        print(f"    {name:6s}: {len(loader.dataset):6d} samples")

    return loaders


def cmd_info(args):
    """Print device and environment info."""
    info = device_info()
    for k, v in info.items():
        print(f"  {k:20s}: {v}")


def cmd_train(args):
    """Run training with curriculum learning on the DOT dataset."""
    from training.engine import Trainer
    from training.curriculum import CurriculumScheduler, Stage
    from training.schedulers import build_scheduler
    from utils.logger import ExperimentLogger
    from data.unify import num_classes

    cfg = Config()
    cfg.model_name = args.model
    device = get_device()
    n_cls = num_classes()

    # Determine epochs
    if args.epochs:
        epochs = args.epochs
    elif args.stage == "geometric":
        epochs = cfg.train.epochs_stage1
    elif args.stage == "real_world":
        epochs = cfg.train.epochs_stage2
    else:
        epochs = cfg.train.epochs_stage1 + cfg.train.epochs_stage2

    # Build model
    model = build_model(cfg.model_name, n_cls)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr or cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = build_scheduler(optimizer, name=cfg.train.scheduler, total_epochs=epochs)

    # Build logger
    logger = ExperimentLogger(
        log_dir=LOG_DIR,
        function="train",
        model_name=cfg.model_name,
        verbose=args.verbose,
    )

    # Per-model checkpoint directory
    model_ckpt_dir = CHECKPOINT_DIR / cfg.model_name
    model_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Build trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        logger=logger,
        use_amp=cfg.train.use_amp,
        checkpoint_dir=model_ckpt_dir,
        save_best=True,
        save_every_n_epochs=args.save_every,
        early_stopping_patience=args.patience,
    )

    # Resume from checkpoint if --continue is set
    start_epoch = 1
    if getattr(args, 'continue_training', False):
        # Find best checkpoint for this model
        ckpt_candidates = [
            model_ckpt_dir / "checkpoint_last.pth",
            model_ckpt_dir / "best_model.pth",
        ]
        # Also check for periodic checkpoints (e.g., checkpoint_epoch_10.pth)
        periodic = sorted(model_ckpt_dir.glob("checkpoint_epoch_*.pth"),
                          key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 0,
                          reverse=True)
        ckpt_candidates = periodic + ckpt_candidates  # newest periodic first

        resumed = False
        for ckpt_path in ckpt_candidates:
            if ckpt_path.exists():
                try:
                    loaded_epoch = trainer.load_checkpoint(ckpt_path)
                    start_epoch = loaded_epoch + 1
                    print(f"\n🔄 Resuming from {ckpt_path.name} (epoch {loaded_epoch})")
                    print(f"   Previous best {trainer.best_metric}: {trainer.best_value:.6f} @ epoch {trainer.best_epoch}")
                    resumed = True
                    break
                except RuntimeError as e:
                    if "state_dict" in str(e) or "Missing key" in str(e):
                        print(f"   ⚠️  {ckpt_path.name}: architecture mismatch, skipping")
                        continue
                    raise

        if not resumed:
            print(f"\n⚠️  No compatible checkpoint found in {model_ckpt_dir} — starting fresh")

    # Build data loaders
    loaders = build_dot_loaders(batch_size=args.batch_size)

    # When continuing, --epochs means "N more epochs", not total target
    if start_epoch > 1:
        remaining_epochs = epochs  # train N more epochs from where we left off
        target_epoch = start_epoch + remaining_epochs - 1
    else:
        remaining_epochs = epochs
        target_epoch = epochs

    print(f"\n🚀 Training {cfg.model_name} for {remaining_epochs} epochs on {device}")
    if start_epoch > 1:
        print(f"   Continuing from epoch {start_epoch} → {target_epoch}")
    print(f"   Stage: {args.stage}")
    print(f"   LR: {args.lr or cfg.train.lr}, Batch: {args.batch_size}")
    print(f"   Checkpoints: {model_ckpt_dir}")
    if args.verbose and logger.log_file_path:
        print(f"   Log file: {logger.log_file_path}")
    print()

    # Train
    history = trainer.fit(loaders["train"], loaders["val"],
                          epochs=remaining_epochs, start_epoch=start_epoch)

    # Summary
    print(f"\n{'─' * 50}")
    print(f"✅ Training complete!")
    print(f"   Best {trainer.best_metric}: {trainer.best_value:.6f} @ epoch {trainer.best_epoch}")
    print(f"   Final train_loss: {history['train_loss'][-1]:.4f}")
    print(f"   Final val_loss:   {history['val_loss'][-1]:.4f}")
    if trainer.stopped_early:
        print(f"   ⏹ Stopped early at epoch {len(history['train_loss'])}")
    print(f"   Checkpoints saved to: {model_ckpt_dir}")
    if args.verbose and logger.log_file_path:
        print(f"   Verbose log: {logger.log_file_path}")

    logger.close()


def cmd_evaluate(args):
    """Run evaluation on a split."""
    import numpy as np
    from data.unify import num_classes
    from utils.logger import ExperimentLogger
    from evaluation.metrics import collect_predictions, classification_report_dict, critical_class_recall
    from evaluation.calibration import compute_ece

    device = get_device()
    n_cls = num_classes()
    model = build_model(args.model, n_cls)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"❌ Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
            print(f"📦 Loaded checkpoint: {ckpt_path} (epoch {state.get('epoch', '?')})")
        else:
            model.load_state_dict(state)
            print(f"📦 Loaded weights: {ckpt_path}")
    else:
        print("⚠️  No checkpoint specified — evaluating with random weights")

    model.to(device).eval()

    # Build loaders
    loaders = build_dot_loaders(batch_size=args.batch_size)

    # Logger
    logger = ExperimentLogger(
        log_dir=LOG_DIR, function="eval",
        model_name=args.model, verbose=args.verbose,
    )

    # Evaluate
    split = args.split
    print(f"\n📊 Evaluating on {split.upper()} split ({device})")

    labels, preds, probs = collect_predictions(model, loaders[split], device)

    # Classification report
    report = classification_report_dict(labels, preds)
    logger.log_evaluation(split, report)

    acc = report.get("accuracy", 0)
    print(f"\n   Accuracy: {acc:.4f}")

    # Per-class detail (if scikit-learn available)
    macro = report.get("macro avg", {})
    if macro:
        print(f"   Macro F1: {macro.get('f1-score', 0):.4f}")
        print(f"   Macro Precision: {macro.get('precision', 0):.4f}")
        print(f"   Macro Recall: {macro.get('recall', 0):.4f}")

    # Critical class recall
    present = sorted(set(labels.tolist()))
    critical = [idx for idx in [0, 1, 18] if idx in present]
    if critical:
        recalls = critical_class_recall(labels, preds, critical_indices=critical)
        print(f"\n   Critical class recall:")
        for name, val in recalls.items():
            print(f"     {name:20s}: {val:.4f}")

    # ECE
    ece, _, _, _ = compute_ece(labels, probs, n_bins=10)
    print(f"\n   ECE: {ece:.4f}")

    if args.verbose and logger.log_file_path:
        print(f"\n   Verbose log: {logger.log_file_path}")

    logger.close()


# ===================================================================
# compare — Classical ML comparison on CNN features
# ===================================================================

def cmd_compare(args):
    """Extract CNN features and compare classical ML models."""
    from data.unify import num_classes
    from evaluation.classical_ml import run_comparison

    device = get_device()
    n_cls = num_classes()
    model = build_model(args.model, n_cls)

    # Auto-find or use specified checkpoint
    ckpt_path = Path(args.checkpoint) if args.checkpoint else CHECKPOINT_DIR / args.model / "best_model.pth"
    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        print(f"   Train the model first: python main.py train --model {args.model}")
        sys.exit(1)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        print(f"📦 Loaded {args.model} checkpoint: {ckpt_path} (epoch {state.get('epoch', '?')})")
    else:
        model.load_state_dict(state)
        print(f"📦 Loaded weights: {ckpt_path}")

    model.to(device).eval()

    # Build loaders
    loaders = build_dot_loaders(batch_size=args.batch_size)

    print(f"\n🔬 Classical ML Comparison (features from {args.model})")
    print(f"   Training on train split, evaluating on {args.split} split\n")

    report = run_comparison(
        model=model,
        train_loader=loaders["train"],
        test_loader=loaders[args.split],
        model_name=args.model,
        device=device,
        verbose=True,
    )

    print(f"\n{report.summary_table()}")
    print(f"\n   Feature dim: {report.feature_dim}")
    print(f"   Train samples: {report.n_train}, Test samples: {report.n_test}")
    print(f"   Classes: {report.n_classes}")


def main():
    parser = argparse.ArgumentParser(
        description="Traffic Sign Recognition — Monolithic CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # info
    sub.add_parser("info", help="Show device/environment info")

    # train
    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--model", default="resnet10",
                         choices=["baseline", "advanced", "resnet10", "orion"])
    p_train.add_argument("--stage", default="full",
                         choices=["geometric", "real_world", "full"])
    p_train.add_argument("--epochs", type=int, default=None,
                         help="Override total epochs (default: from config)")
    p_train.add_argument("--lr", type=float, default=None,
                         help="Learning rate (default: 1e-3)")
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--patience", type=int, default=0,
                         help="Early stopping patience (0=disabled)")
    p_train.add_argument("--save-every", type=int, default=5,
                         help="Save checkpoint every N epochs")
    p_train.add_argument("--verbose", action="store_true",
                         help="Enable verbose file logging")
    p_train.add_argument("--continue", dest="continue_training",
                         action="store_true",
                         help="Resume training from last/best checkpoint")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate a model")
    p_eval.add_argument("--model", default="resnet10",
                        choices=["baseline", "advanced", "resnet10", "orion"])
    p_eval.add_argument("--split", default="test",
                        choices=["val", "test", "ood"])
    p_eval.add_argument("--checkpoint", default=None)
    p_eval.add_argument("--batch-size", type=int, default=8)
    p_eval.add_argument("--verbose", action="store_true",
                         help="Enable verbose file logging")

    # compare
    p_comp = sub.add_parser("compare", help="Compare classical ML models on CNN features")
    p_comp.add_argument("--model", default="resnet10",
                        choices=["baseline", "advanced", "resnet10", "orion"])
    p_comp.add_argument("--split", default="test",
                        choices=["val", "test", "ood"])
    p_comp.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint (default: auto-find best)")
    p_comp.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()
    if args.command == "info":
        cmd_info(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
