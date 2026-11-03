from pathlib import Path
from typing import Sequence, Union

import tensorflow as tf


def decode_and_resize_image(
    image_path: Union[str, Path, tf.Tensor],
    image_size: Sequence[int],
) -> tf.Tensor:
    if isinstance(image_path, Path):
        image_path = str(image_path)

    image = tf.io.decode_image(
        tf.io.read_file(image_path),
        channels=3,
        expand_animations=False,
    )
    image.set_shape((None, None, 3))
    return tf.image.resize(image, tuple(image_size[:2]))
