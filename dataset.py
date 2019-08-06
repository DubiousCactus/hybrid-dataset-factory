#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Theo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the GPLv3 license.

"""
Dataset class, holding background images along with their annotations
"""

import random
import json
import os

from PIL import Image
from tqdm import tqdm
from queue import Queue
from threading import Thread
from pyrr import Vector3, Quaternion


class BackgroundAnnotations:
    def __init__(self, translation: Vector3, orientation: Quaternion):
        self.translation = translation
        self.orientation = orientation


'''
Holds a background image along with its annotations
'''
class BackgroundImage:
    def __init__(self, image_path: str, annotations: BackgroundAnnotations):
        self.file = image_path
        self.annotations = annotations

    def image(self):
        return Image.open(self.file)


class SyntheticAnnotations:
    def __init__(self, bboxes):
        self.bboxes = bboxes


'''
Holds a generated image along with its annotations
'''
class AnnotatedImage:
    def __init__(self, image: Image, id, annotations: SyntheticAnnotations):
        self.image = image
        self.id = id
        self.annotations = annotations


class Dataset:
    def __init__(self, path: str, seed=None, max=0):
        if not os.path.isdir(path):
            raise Exception("Dataset directory {} not found".format(path))
        if seed:
            random.seed(seed)
        else:
            random.seed()
        self.path = path
        self.width = None
        self.height = None
        self.data = Queue(maxsize=max)
        self.saving = False

    def parse_annotations(self, path: str):
        if not os.path.isfile(path):
            raise Exception("Annotations file not found")
        annotations = dict()
        with open(path) as file:
            file.readline() # Discard the header
            for line in file:
                items = line.split(',')
                annotations[items[0].strip()] = BackgroundAnnotations(
                    Vector3([float(x) for x in items[1:4]]),
                    Quaternion([float(x) for x in items[4:8]])
                )

        return annotations

    def load(self, count, annotations_path=None, randomize=True):
        print("[*] Loading and randomizing base dataset...")
        if randomize:
            files = os.listdir(self.path)
            random.shuffle(files)
        else:
            file = sorted(os.listdir(self.path))

        annotations = self.parse_annotations(annotations_path)
        # Remove files without annotations
        files = [file for file in files if file in annotations]
        while count > len(files):
            choice = random.choice(files)
            full_path = os.path.join(self.path, choice)
            if os.path.isfile(full_path) and full_path != annotations_path:
                files += [choice]

        for file in files:
            full_path = os.path.join(self.path, file)
            if os.path.isfile(full_path) and full_path != annotations_path:
                self.data.put(BackgroundImage(full_path, annotations[file]))
                self.data.task_done()
                if not self.width and not self.height:
                    with Image.open(full_path) as img:
                        self.width, self.height = img.size

        self.data.join()
        return self.data.qsize() != 0

    '''
    Returns the next BackgroundImage in the Queue
    '''
    def get(self):
        return self.data.get()

    def task_done(self):
        self.data.task_done()

    def put(self, image: AnnotatedImage):
        self.data.put(image)

    # Runs in a thread
    def save(self):
        if not self.saving:
            with open(os.path.join(self.path, 'annotations.json'),
                      'w', encoding='UTF-8') as f:
                annotations = {}
                annotations['classes'] = [
                    {'id': 0, 'label': 'Background'},
                    {'id': 1, 'label': 'Target 1'},
                    {'id': 2, 'label': 'Target 2'},
                    {'id': 3, 'label': 'Forward gate'},
                    {'id': 4, 'label': 'Backward gate'}
                ]
                annotations['annotations'] = []
                json.dump(annotations, f, ensure_ascii=False, indent=4)

            self.saving = True
            if not os.path.isdir(os.path.join(self.path, 'images')):
                os.mkdir(os.path.join(self.path, 'images'))

        for annotatedImage in iter(self.data.get, None):
            name = "%06d.png" % annotatedImage.id
            annotatedImage.image.save(
                os.path.join(self.path, 'images', name)
            )
            bboxes = []
            for bbox in annotatedImage.annotations.bboxes:
                bboxes.append({
                    'class_id': bbox['class_id'],
                    'xmin': bbox['min'][0],
                    'ymin': bbox['min'][1],
                    'xmax': bbox['max'][0],
                    'ymax': bbox['max'][1],
                    'distance': bbox['distance'],
                    'rotation': bbox['rotation']
                })

            annotation = {
                'image': name,
                'annotations': bboxes
            }

            with open(os.path.join(self.path, 'annotations.json'),
                      'r', encoding='UTF-8') as f:
                data = json.load(f)

            data['annotations'].append(annotation)

            with open(os.path.join(self.path, 'annotations.json'),
                      'w', encoding='UTF-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    def get_image_size(self):
        print("[*] Using {}x{} base resolution".format(self.width, self.height))
        return (self.width, self.height)
