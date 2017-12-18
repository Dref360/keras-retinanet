import json
import os
from collections import defaultdict

import cv2

pjoin = os.path.join

bbox_result = json.load(open('scripts/test_bbox_results.json', 'r'))
img_ids = json.load(open('scripts/test_processed_image_ids.json', 'r'))
ROOT = '/media/braf3002/hdd2/Downloads/MIO-TCD-Localization/test'
print(bbox_result)
print(img_ids)

all_data = defaultdict(list)

for img_data in bbox_result:
    img_id = img_data['image_id'] + '.jpg'
    all_data[img_id].append((img_data['category_id'], img_data['score'], img_data['bbox']))

acc = []
for fp, boxes in all_data.items():
    #img = cv2.imread(pjoin(ROOT, fp))
    for cls, sc, (x, y, w, h) in boxes:
        #cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 1)
        acc.append([fp[:-4], cls, sc, x, y, x + w, y + h])
    #cv2.imshow('machin', img)
    #cv2.waitKey(1000)

with open('result.csv','w') as f:
    f.writelines([','.join([str(k) for k in line])+'\n' for line in acc])
