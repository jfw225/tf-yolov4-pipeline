class REDIS:
    REDIS_HOST = "127.0.0.1"
    REDIS_PORT = 6379


class MODEL:

    class SAVE:
        WEIGHTS = "data/yolov4.weights"
        CLASSES = "data/classes/coco.names"
        FRAMEWORK = "tf"
        MODEL = "yolov4"
        IMAGE_SIZE = 416
        SCORE_THRESH = 0.2
        OUTPUT = None

    class EVAL:
        INPUT = "data/images/"
        WEIGHTS = "checkpoints/yolov4-416-test"
        CLASSES = "data/classes/coco.names"
        FRAMEWORK = "tf"
        IMAGE_SIZE = 416
        IOU_THRESH = 0.45
        SCORE_THRESH = 0.25
