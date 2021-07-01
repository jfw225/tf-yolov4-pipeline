# tf-yolov4-pipeline
 
Modular image processing pipeline implementation of [tensorflow-yolov4-tflite] (https://github.com/hunglc007/tensorflow-yolov4-tflite).


## Setup environment

This repo implements another repo: [tfpipe] (https://github.com/jfw225/tfpipe). To make sure that it is installed, do the following: 

    $ git submodule update --init --recursive

Next, use your preferred package manager to install the packages listed in `requirements.txt`.

Using pip:

    $ python -m pip install -r requirements.txt

## Command Line Arguments and Config

Each script (`save_model.py` and `process_images.py`) has a set of command line arguments. You can see a full list of these arguments by running:

    $ python save_model.py --help
    $ python process_images.py --help

However, each of these parameters can also be changed in the `config.py` file.

## Converting from Darknet

First, change the config or set the parameters for your `.weights` file. Next, execute: 

    $ python save_model.py

## Basic Usage

Install the appropriate model, config, data, and class files. You can then use `process_images.py` to process images.

Run the following command to see all of the available options:

    $ python process_images.py --help

The `-i` or `--input` arguments are used to specify an input file path. The file path can either be a directory or a file, and the types of files that can be read are specified in `config.py`.

Here is a demo call using the the COCO model from the repo linked above:

    $ python process_images.py -w checkpoints/yolov4-416 -i data/


## Redis

To use Redis, you must specify the `--redis` argument. The host and port of the Redis server can be specified by `--redis-host` and `--redis-port` arguments.

To submit images to the processing queue, open your `redis-cli` and publish a file path to the `frames` channel:

    $ publish frames <image/directory file path>

To read the output of the model, have a seperate `redis-cli` subscribe to the `bbox` channel:

    $ subscribe bbox



## TODO

add libdevice.10.bc
use cpu for non-predict functions in async
cpus corresponding to gpus
wrap more things in tf.function


figure out appropriate vram
create multiple virtual gpus
make sure the only gpu operations are in prediction
get rid of superfluous keys in data

write get_preproc function

create virtual cpus

change all to use tf functions (update redis)
wrap preproc in tf function
wrap annotate in tf function

abstract bounding box iterators

abstract redis input and image input

try xlagpu https://stackoverflow.com/questions/52943489/what-is-xla-gpu-and-xla-cpu-for-tensorflow