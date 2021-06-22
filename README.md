# tf-yolov4-pipeline
 
Modular image processing pipeline implementation of [tensorflow-yolov4-tflite] (https://github.com/hunglc007/tensorflow-yolov4-tflite).


## Setup environment

Use your preferred package manager to install the packages listed in `requirements.txt`.

Using pip:

    $ python -m pip install -r requirements.txt

## Basic Usage

Install the appropriate model, config, data, and class files. You can then use `process_image.py` to process images.

Run the following command to see all of the available options:

    $ python process_image.py --help

The `-i` or `--input` arguments are used to specify an input file path. The file path can either be a directory or a file, and the types of files that can be read are specified in `config.py`.

Here is a demo call using the the COCO model from the repo linked above:

    $ python process_image.py -w checkpoints/yolov4-416 -i data/


## Redis

To use Redis, you must specify the `--redis` argument. The host and port of the Redis server can be specified by `--redis-host` and `--redis-port` arguments.

To submit images to the processing queue, open your `redis-cli` and publish a file path to the `frames` channel:

    $ publish frames <image/directory file path>

To read the output of the model, have a seperate `redis-cli` subscribe to the `bbox` channel:

    $ subscribe bbox



## TODO

add libdevice.10.bc
deal with bilinear issue
write model convertor from darknet weights
do __call__ with pipe
use cpu for non-predict functions in async
cpus corresponding to gpus