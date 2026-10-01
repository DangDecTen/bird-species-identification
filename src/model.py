"""Model definitions."""
import keras

BACKBONES = {
    "mobilenetv3small": keras.applications.MobileNetV3Small,
    "mobilenetv3large": keras.applications.MobileNetV3Large,
}


def build_model(backbone, num_classes, img_size=224, dropout=0.2, freeze_backbone=True):
    """Pretrained backbone + softmax head.

    Input: float32 RGB in [0, 255]. MobileNetV3 has its preprocessing built in
    (include_preprocessing=True), so the app does NOT need mean/std normalization.
    Output: softmax probabilities (num_classes).
    """
    base = BACKBONES[backbone](
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
        include_preprocessing=True,
    )
    base.trainable = not freeze_backbone

    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = base(inputs, training=False)  # keep BatchNorm in inference mode
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return keras.Model(inputs, outputs, name=f"bird_{backbone}")
