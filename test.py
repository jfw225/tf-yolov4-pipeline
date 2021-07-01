import tensorflow as tf

from tfpipe.pipeline.image_input import ImageInput

INPUT_PATH = "data/images/"
MODEL_SIZE = 416
META = False


def main():
    image_input = ImageInput(
        path=INPUT_PATH, size=MODEL_SIZE, meta=META)
