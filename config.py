class REDIS:
    HOST = "127.0.0.1"
    PORT = 6379
    CH_IN = ""
    CH_OUT = ""


class MODEL:

    class SAVE:
        WEIGHTS = "data/yolov4.weights"
        CLASSES = "data/classes/coco.names"
        FRAMEWORK = "tf"
        MODEL = "yolov4"
        IMAGE_SIZE = 416
        SCORE_THRESH = 0.2
        OUTPUT = "checkpoints/yolov4-416"

    class EVAL:
        INPUT = "data/images/"
        WEIGHTS = "checkpoints/yolov4-416"
        CLASSES = "data/classes/coco.names"
        FRAMEWORK = "tf"
        IMAGE_SIZE = 416
        IOU_THRESH = 0.45
        SCORE_THRESH = 0.25
