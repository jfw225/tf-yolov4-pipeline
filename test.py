import re
from tfpipe.pipeline.pipeline import Pipeline
import tensorflow as tf

from tfpipe.pipeline.pipeline import Pipeline
from tfpipe.pipeline.image_input import ImageInput

from time import time

INPUT_PATH = "data/images/"
MODEL_SIZE = 416
META = False


def main():

    # GPU Logging
    tf.debugging.set_log_device_placement(False)

    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)

    # Create Pipeline Tasks
    image_input = ImageInput(
        path=INPUT_PATH, size=MODEL_SIZE, meta=META)

    pipeline = image_input

    # Main Loop
    t = time()
    index = 0
    results = list()
    while image_input.is_working():
        result = pipeline(None)
        if result != Pipeline.Skip:
            results.append(result)

            index += 1

    runtime = time() - t
    print(
        f"Images Processed: {index} imgs | Runtime: {runtime} s | Rate: {index / runtime} imgs/s")

    image_input.cleanup()

    # print(results)


if __name__ == '__main__':
    main()
