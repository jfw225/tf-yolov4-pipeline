class REDIS:
    REDIS_HOST = "127.0.0.1"
    REDIS_PORT = 6379


class MODEL:

    class SAVE:
        WEIGHTS = "data/fpsv2.weights"
        CLASSES = "data/fps.names"
        FRAMEWORK = "tf"
        MODEL = "yolov4"
        IMAGE_SIZE = 416
        SCORE_THRESH = 0.2
        OUTPUT = "checkpoints/fpsv2-yolov4-416"

    class EVAL:
        INPUT = "data/images/"
        WEIGHTS = "checkpoints/yolov4-416-m"
        CLASSES = "data/classes/coco.names"
        FRAMEWORK = "tf"
        IMAGE_SIZE = 416
        IOU_THRESH = 0.45
        SCORE_THRESH = 0.25
