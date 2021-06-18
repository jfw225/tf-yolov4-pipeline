class REDIS:
    REDIS_HOST = "127.0.0.1"
    REDIS_PORT = 6379


class MODEL:
    FRAMEWORK = "tf"
    WEIGHTS = "checkpoints/yolov4-416"
    CLASSES = "data/classes/coco.names"
    IMAGE_SIZE = 416
    IOU_THRESH = 0.45
    SCORE_THRESH = 0.25
