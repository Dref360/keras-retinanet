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

from __future__ import print_function

import numpy as np
import json
import os
from keras_retinanet.cython_utils.nms import SNMS

from tqdm import tqdm


def evaluate_csv(generator, model, threshold=0.1):
    # start collecting results
    results = []
    image_ids = []
    print("Start evaluation")
    try:
        for i in tqdm(range(generator.size())):
            image = generator.load_image(i)
            image = generator.preprocess_image(image)
            image, scale = generator.resize_image(image)

            # run network
            _, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))

            # clip to image shape
            detections[:, :, 0] = np.maximum(0, detections[:, :, 0])
            detections[:, :, 1] = np.maximum(0, detections[:, :, 1])
            detections[:, :, 2] = np.minimum(image.shape[1], detections[:, :, 2])
            detections[:, :, 3] = np.minimum(image.shape[0], detections[:, :, 3])

            # correct boxes for image scale
            detections[0, :, :4] /= scale

            # change to (x, y, w, h) (MS COCO standard)
            detections[:, :, 2] -= detections[:, :, 0]
            detections[:, :, 3] -= detections[:, :, 1]

            # compute predicted labels and scores
            boxes = SNMS(detections[0][:,4:].copy(order='C'),detections[0][:,:4].copy(order='C'))
            for box in boxes:
                positive_labels = np.where(box.probs > threshold)[0]

                # append detections for each positively labeled class
                for label in positive_labels:
                    image_result = {
                        'image_id'    : generator.image_names[i],
                        'category_id' : generator.label_to_name(label),
                        'score'       : float(box.probs[label]),
                        'bbox'        : [box.x,box.y,box.w,box.h],
                    }

                    # append detection to results
                    results.append(image_result)
            # append image to list of processed images
            image_ids.append(generator.image_names[i])

            # print progress
            #print('{}/{}'.format(i, len(generator.image_ids)), end='\r')

        if not len(results):
            return
    except KeyboardInterrupt:
        pass

    # write output
    json.dump(results, open('test_bbox_results.json', 'w'), indent=4)
    json.dump(image_ids, open('test_processed_image_ids.json', 'w'), indent=4)

