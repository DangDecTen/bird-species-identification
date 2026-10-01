"""Evaluate a saved model on a split (default: test) and save diagnostics.

Usage:  python -m src.evaluate --run-name b0_mnv3small_frozen
Look at the TEST split sparingly: use val for model selection.
"""
import argparse
import json
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.data import load_split, prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--split-dir", default="data/splits")
    a = p.parse_args()

    model = keras.saving.load_model(f"models/{a.run_name}.keras")
    img_size = model.input_shape[1]
    ds, classes = load_split(a.split_dir, a.split, 32, img_size, shuffle=False)

    y_true = np.concatenate([y.numpy() for _, y in ds])
    probs = model.predict(prepare(ds), verbose=0)
    pred = probs.argmax(1)
    top3_idx = np.argsort(-probs, axis=1)[:, :3]

    acc = float((pred == y_true).mean())
    top3 = float(np.mean([t in row for t, row in zip(y_true, top3_idx)]))
    cm = confusion_matrix(y_true, pred, labels=range(len(classes)))

    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(
        ((int(off[i, j]), classes[i], classes[j]) for i, j in zip(*np.nonzero(off))), reverse=True
    )[:5]

    out = Path("experiments") / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.split}_metrics.json").write_text(
        json.dumps({"n": len(y_true), "acc": acc, "top3": top3, "top_confusions(count,true,pred)": pairs}, indent=2)
    )
    report = classification_report(
        y_true, pred, labels=range(len(classes)), target_names=classes, digits=3, zero_division=0
    )
    (out / f"{a.split}_report.txt").write_text(report)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.tight_layout(); fig.savefig(out / f"{a.split}_confusion_matrix.png", dpi=120); plt.close(fig)

    results = Path("experiments/results.csv")
    if results.exists():
        df = pd.read_csv(results)
        df.loc[df.run_name == a.run_name, [f"{a.split}_acc", f"{a.split}_top3"]] = [round(acc, 4), round(top3, 4)]
        df.to_csv(results, index=False)

    print(report)
    print(f"{a.split}: n={len(y_true)}  acc={acc:.3f}  top3={top3:.3f}")
    print("most confused (count, true, predicted):", *pairs, sep="\n  ")


if __name__ == "__main__":
    main()
