#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
import keras.preprocessing.image
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.utils.csv_eval import evaluate_csv
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.keras_version import check_keras_version

import tensorflow as tf

import argparse
import os


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def parse_args():
    parser = argparse.ArgumentParser(description='Simple training script for COCO object detection.')
    parser.add_argument('model', help='Path to RetinaNet model.')
    parser.add_argument('csv', help='Path to CSV')
    parser.add_argument('cls_file', help='Path to CSV cls')
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold', help='Threshold on score to filter detections with (defaults to 0.05).', default=0.1, type=float)

    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the model
    print('Loading model, this may take a second...')
    model = keras.models.load_model(args.model, custom_objects=custom_objects)

    # create image data generator object
    test_image_data_generator = keras.preprocessing.image.ImageDataGenerator()

    # create a generator for testing data
    test_generator = CSVGenerator(args.csv, args.cls_file,test_image_data_generator)

    evaluate_csv(test_generator, model, args.score_threshold)
