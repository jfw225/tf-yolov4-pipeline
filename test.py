from tfpipe.pipeline.annotate_image import AnnotateImage
from tfpipe.pipeline.pipeline import Pipeline
import tensorflow as tf

from tfpipe.pipeline.pipeline import Pipeline
from tfpipe.pipeline.image_input import ImageInput

from time import time
import pickle

INPUT_PATH = "data/images/"
MODEL_SIZE = 416
META = False
CLASSES = "data/classes/coco.names"
IOU_THRESH = 0.45
SCORE_THRESH = 0.25
OUTPUT_TYPE = "vis_image"


def main():

    # GPU Logging
    tf.debugging.set_log_device_placement(False)

    # # Create Pipeline Tasks
    # image_input = ImageInput(
    #     path=INPUT_PATH, size=MODEL_SIZE, meta=META)

    annotate_image = AnnotateImage(
        OUTPUT_TYPE, IOU_THRESH, SCORE_THRESH, META, CLASSES)

    pipeline = annotate_image  # image_input

    data = pickle.load(open("test.pkl", "rb"))

    # Main Loop
    t = time()
    index = 0
    results = list()
    # while image_input.is_working():
    for _ in range(1000):
        result = pipeline(data)
        if result != Pipeline.Skip:
            # results.append(result)

            index += 1

    runtime = time() - t
    print(
        f"Images Processed: {index} imgs | Runtime: {runtime} s | Rate: {index / runtime} imgs/s")

    # image_input.cleanup()

    # print(results)


if __name__ == '__main__':
    main()
