"""Train a model and log validation results.

Usage (from repo root):
    python -m src.train --run-name b0_mnv3small_frozen
"""
import argparse
from datetime import datetime
from pathlib import Path

import keras
import pandas as pd

from src.data import load_split, prepare
from src.model import build_model

RESULTS = Path("experiments/results.csv")


def log_result(row):
    RESULTS.parent.mkdir(exist_ok=True)
    new = pd.DataFrame([row])
    df = pd.concat([pd.read_csv(RESULTS), new], ignore_index=True) if RESULTS.exists() else new
    df.to_csv(RESULTS, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--backbone", default="mobilenetv3small")
    p.add_argument("--split-dir", default="data/splits")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    a = p.parse_args()

    keras.utils.set_random_seed(a.seed)

    train_ds, classes = load_split(a.split_dir, "train", a.batch_size, a.img_size, seed=a.seed)
    val_ds, val_classes = load_split(a.split_dir, "val", a.batch_size, a.img_size, shuffle=False)
    assert classes == val_classes, "train/val class folders differ"

    model = build_model(a.backbone, len(classes), a.img_size, a.dropout)
    model.compile(
        optimizer=keras.optimizers.Adam(a.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3")],
    )
    hist = model.fit(
        prepare(train_ds, augment=not a.no_augment),
        validation_data=prepare(val_ds),
        epochs=a.epochs,
        callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True)],
    )

    val = model.evaluate(prepare(val_ds), return_dict=True, verbose=0)  # best weights restored
    Path("models").mkdir(exist_ok=True)
    model.save(f"models/{a.run_name}.keras")

    log_result(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_name": a.run_name,
            "backbone": a.backbone,
            "augment": not a.no_augment,
            "lr": a.lr,
            "batch_size": a.batch_size,
            "dropout": a.dropout,
            "img_size": a.img_size,
            "epochs_run": len(hist.history["loss"]),
            "val_loss": round(val["loss"], 4),
            "val_acc": round(val["accuracy"], 4),
            "val_top3": round(val["top3"], 4),
        }
    )
    print(f"\nval_acc={val['accuracy']:.3f}  val_top3={val['top3']:.3f}  -> models/{a.run_name}.keras")


if __name__ == "__main__":
    main()
