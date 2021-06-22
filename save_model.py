import os
from tfpipe.core import config

import config as cfg

from tfpipe.pipeline.create_model import CreateModel
from tfpipe.pipeline.convert_weights import ConvertWeights


def parse_args():
    """ Parses command line arguments. """

    import argparse

    # Parse command line arguments
    ap = argparse.ArgumentParser(
        description="Darknet to TensorFlow Model Conversion Pipeline")
    ap.add_argument("-w", "--weights", default=cfg.MODEL.SAVE.WEIGHTS,
                    help="path to weights file")
    ap.add_argument("-s", "--size", type=int, default=cfg.MODEL.SAVE.IMAGE_SIZE,
                    help="the value to which the images will be resized")

    # Model Settings
    ap.add_argument("-f", "--framework", default=cfg.MODEL.SAVE.FRAMEWORK,
                    help="the framework of the model")
    ap.add_argument("--tiny", action="store_true",
                    help="use yolo-tiny instead of yolo")
    ap.add_argument("--model", default=cfg.MODEL.SAVE.MODEL,
                    help="yolov4 or yolov3")
    ap.add_argument("--score", default=cfg.MODEL.SAVE.SCORE_THRESH,
                    help="score threshold")
    ap.add_argument("--classes", default=cfg.MODEL.SAVE.CLASSES,
                    help="file path to classes")

    # Output Settings
    ap.add_argument("-o", "--output", default=cfg.MODEL.SAVE.OUTPUT,
                    help="path to the output directory")

    return ap.parse_args()


def main(args):

    # Create Pipeline Tasks
    create_model = CreateModel(input_size=args.size,
                               classes=args.classes,
                               framework=args.framework,
                               model_name=args.model,
                               is_tiny=args.tiny)

    if args.output is None:
        os.makedirs("checkpoints", exist_ok=True)
        args.output = f"checkpoints/{args.model}-{args.size}"

    convert_weights = ConvertWeights(weights=args.weights,
                                     output=args.output,
                                     model_name=args.model,
                                     is_tiny=args.tiny)

    # Create Pipeline
    pipeline = create_model >> convert_weights

    # Complete Task
    pipeline.map(None)


if __name__ == '__main__':
    args = parse_args()
    main(args)
