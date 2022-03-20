import cv2
import numpy as np
from typing import *
from skimage.measure import find_contours


classes = ['apple', 'banana', 'bread', 'bread-black', 'chocolate', 'fish',
           'juice-orange', 'kiwi', 'mandarine', 'milk', 'orange', 'rice', 'tea-black', 'tea-green']

# cm, ccal, g/cm^3, cm^3
cals_table = {
    'apple': {
        "volume": 523,
        "z_real": 7,
        "density": 0.7,
        "cal_per_100": 52
    },
    'banana': {
        "volume": 180,
        "z_real": 3,
        "density": 0.75,
        "cal_per_100": 89
    },
    'bread': {
        "volume": 122,
        "z_real": 9,
        "density": 1.31,
        "cal_per_100": 265
    },
    'bread-black': {
        "volume": 122,
        "z_real": 9,
        "density": 1.21,
        "cal_per_100": 259
    },
    'chocolate': {
        "volume": 60,
        "z_real": 0.8,
        "density": 1.2,
        "cal_per_100": 546
    },
    'fish': {
        "volume": 168,
        "z_real": 2,
        "density": 0.98,
        "cal_per_100": 142
    },
    'juice-orange': {
        "volume": 402,
        "z_real": 8,
        "density": 1.043,
        "cal_per_100": 45
    },
    'kiwi': {
        "volume": 500,
        "z_real": 5,
        "density": 0.8,
        "cal_per_100": 48
    },
    'mandarine': {
        "volume": 520,
        "z_real": 5,
        "density": 1.1,
        "cal_per_100": 38
    },
    'milk': {
        "volume": 402,
        "z_real": 8,
        "density": 1.027,
        "cal_per_100": 64
    },
    'orange': {
        "volume": 3052,
        "z_real": 9,
        "density": 1.1,
        "cal_per_100": 38
    },
    'rice': {
        "volume": 1000,
        "z_real": 2,
        "density": 0.7,
        "cal_per_100": 344
    },
    'tea-black': {
        "volume": 402,
        "z_real": 8,
        "density": 1,
        "cal_per_100": 0
    },
    'tea-green': {
        "volume": 402,
        "z_real": 8,
        "density": 1,
        "cal_per_100": 0
    }
}


def eucledian(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**.5


def PolyArea(x, y):
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


class calories:
    def __init__(self, image: np.ndarray, r1, class_names: List[str], protoFile="hand/pose_deploy.prototxt", weightsFile="hand/pose_iter_102000.caffemodel"):
        self.image = image
        self.r1 = r1
        self.class_names = class_names

        self.nPoints = 22
        self.POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [
            10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
        self.threshold = 0.2

        self.width = image.shape[1]
        self.height = image.shape[0]

        self.aspect_ratio = self.width/self.height

        self.inHeight = 368
        self.inWidth = int(((self.aspect_ratio*self.inHeight)*8)//8)

        self.net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    def process(self, hand_size: float = 20):
        dist = self.getHand()
        if dist < 100:
            print("---HAND NOT FOUND. DIVERTING TO BASIC ALGORITHM---")
            return self.countCaloriesBasic()
        else:
            out = {}
            boxes = self.r1['rois']
            class_ids = self.r1['class_ids']
            masks = self.r1['masks']
            scores = self.r1['scores']

            cm_per_px = hand_size / dist

            N = boxes.shape[0]
            s = 0

            for i in range(N):
                if self.class_names[class_ids[i]].lower() == 'orange': continue
                if (not np.any(boxes[i])) or scores[i] < 0.2:
                    continue

                class_id = class_ids[i]
                label = self.class_names[class_id].lower().replace(', ', '')
                if label.lower() == 'orange':
                    continue
                mask = masks[:, :, i]


                padded_mask = np.zeros(
                    (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
                padded_mask[1:-1, 1:-1] = mask
                contours = find_contours(padded_mask, 0.5)

                x = []
                y = []

                for verts in contours:
                    verts = np.fliplr(verts) - 1
                    for i in range(len(verts)):
                        x.append(verts[i][0] * cm_per_px)
                        y.append(verts[i][1] * cm_per_px)

                # hand_real - hand_px
                #     x     -    px

                area = PolyArea(x, y)
                volume = area * cals_table[label]['z_real']
                print(area, volume, dist, cm_per_px,
                      cals_table[label]['density'])
                out[label] = out.get(label, 0) + round(volume * cals_table[label]['density'] *
                                                        cals_table[label]['cal_per_100'] / 100)
                s += round(volume * cals_table[label]['density'] *
                           cals_table[label]['cal_per_100'] / 100)
            out['Сумма'] = s
            out['Тип подсчета калорий'] = 'Рука'
            return out

    def countCaloriesBasic(self):
        out = {}
        boxes = self.r1['rois']
        class_ids = self.r1['class_ids']
        masks = self.r1['masks']
        scores = self.r1['scores']

        N = boxes.shape[0]

        s = 0

        for i in range(N):
            if self.class_names[class_ids[i]].lower() == 'orange': continue
            if (not np.any(boxes[i])) or scores[i] < 0.2:
                continue

            class_id = class_ids[i]
            label = self.class_names[class_id].lower().replace(', ', '')

            out[label] = out.get(label, 0) + round(cals_table[label]['volume'] *
                                                   cals_table[label]['density'] *
                                                   cals_table[label]['cal_per_100'] / 100)
            s += round(cals_table[label]['volume'] *
                       cals_table[label]['density'] *
                       cals_table[label]['cal_per_100'] / 100)
        out['Сумма калорий'] = s
        out['Тип подсчета калорий'] = 'Обычный'
        return out

    def getHand(self) -> float:
        print("---STARTED HAND DETECTION---")
        inpBlob = cv2.dnn.blobFromImage(self.image, 1.0 / 255, (self.inWidth, self.inHeight),
                                        (0, 0, 0), swapRB=False, crop=False)

        self.net.setInput(inpBlob)

        output = self.net.forward()

        points = []

        for i in range(self.nPoints):
            probMap = output[0, i, :, :]
            probMap = cv2.resize(probMap, (self.width, self.height))

            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            if prob > self.threshold:
                points.append((int(point[0]), int(point[1])))
            else:
                points.append(None)

        max_dist = -float('inf')
        print(points)
        for p1 in points:
            for p2 in points:
                if p1 == p2:
                    continue
                if p1 and p2:
                    max_dist = max(eucledian(p1, p2), max_dist)

        return max_dist
