class REDIS:
    REDIS_HOST = "127.0.0.1"
    REDIS_PORT = 6379


class MODEL:
    INPUT = "data/images/"
    FRAMEWORK = "tf"
    WEIGHTS = "checkpoints/yolov4-416-test"
    CLASSES = "data/classes/coco.names"
    IMAGE_SIZE = 416
    IOU_THRESH = 0.45
    SCORE_THRESH = 0.25
