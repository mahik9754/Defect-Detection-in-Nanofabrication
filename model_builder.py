
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam


def build_model(num_classes=6, input_shape=(224, 224, 3)):
    # Load MobileNetV2 pretrained on ImageNet
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained layers initially
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)

    # Convert pixel values from [0, 255] to [-1, 1]
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)

    # Extract image features
    x = base_model(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)

    return model


def compile_model(model, learning_rate=0.001):
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def unfreeze_top(model, num_layers=30):
    # Find the MobileNetV2 base model
    base_model = next(
        layer for layer in model.layers
        if isinstance(layer, Model)
    )

    base_model.trainable = True

    # Freeze all but the last num_layers layers
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    # Keep Batch Normalization layers frozen for stability
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    compile_model(model, learning_rate=1e-5)

    return model
