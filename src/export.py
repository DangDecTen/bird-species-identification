"""Export Keras -> TFLite and check parity against the Keras model.

Usage:  python -m src.export --run-name b0_mnv3small_frozen --modes float32 float16
Writes models/<run>_<mode>.tflite, models/labels.txt, models/model_config.json
"""
import argparse
import json
import re
import tempfile
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from src.data import load_split


def display_name(folder):
    """'141.Artic_Tern' -> 'Arctic Tern' (CUB spells it 'Artic')."""
    return re.sub(r"^\d+\.", "", folder).replace("_", " ").replace("Artic ", "Arctic ")


def convert(model, mode):
    with tempfile.TemporaryDirectory() as tmp:
        model.export(tmp, format="tf_saved_model")  # recommended Keras 3 -> TFLite route
        conv = tf.lite.TFLiteConverter.from_saved_model(tmp)
        if mode == "float16":
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.target_spec.supported_types = [tf.float16]
        elif mode != "float32":
            raise ValueError(mode)
        return conv.convert()


def tflite_predict(tflite_bytes, images):
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    preds = []
    for img in images:
        interp.set_tensor(inp["index"], img[None].astype(np.float32))
        interp.invoke()
        preds.append(interp.get_tensor(out["index"])[0])
    return np.array(preds), inp, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--modes", nargs="+", default=["float32", "float16"])
    p.add_argument("--split-dir", default="data/splits")
    a = p.parse_args()

    model = keras.saving.load_model(f"models/{a.run_name}.keras")
    img_size = model.input_shape[1]
    ds, classes = load_split(a.split_dir, "test", 32, img_size, shuffle=False)
    images = np.concatenate([x.numpy() for x, _ in ds])
    keras_probs = model.predict(images, verbose=0)

    for mode in a.modes:
        tfl = convert(model, mode)
        path = Path(f"models/{a.run_name}_{mode}.tflite")
        path.write_bytes(tfl)
        probs, inp, out = tflite_predict(tfl, images)
        agree = float((probs.argmax(1) == keras_probs.argmax(1)).mean())
        print(f"[{mode}] {path.name}: {path.stat().st_size / 1e6:.2f} MB | "
              f"in={inp['shape'].tolist()} {inp['dtype'].__name__} | out={out['shape'].tolist()} | "
              f"top-1 agreement vs Keras={agree:.3%} | max |dprob|={np.abs(probs - keras_probs).max():.4f}")

    Path("models/labels.txt").write_text("\n".join(display_name(c) for c in classes) + "\n")
    Path("models/model_config.json").write_text(json.dumps({
        "input_size": [img_size, img_size],
        "channel_order": "RGB",
        "input_dtype": "float32",
        "input_range": [0, 255],
        "normalization": "none (built into the model)",
        "resize": "bilinear, stretch to square (no crop, no aspect-ratio padding)",
        "output": "softmax probabilities, same order as labels.txt",
        "num_classes": len(classes),
        "min_confidence": 0.5,
    }, indent=2))
    print("wrote models/labels.txt and models/model_config.json")


if __name__ == "__main__":
    main()
