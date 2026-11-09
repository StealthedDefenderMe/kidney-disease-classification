import tensorflow as tf

model = tf.keras.models.load_model(
    "artifacts/training/best_model_v2.h5"
)

for layer in model.layers:
    print(layer.name, layer.trainable)

    # Inspect nested VGG16 layers
    if isinstance(layer, tf.keras.Model):
        for nested_layer in layer.layers:
            print("   ", nested_layer.name, nested_layer.trainable)