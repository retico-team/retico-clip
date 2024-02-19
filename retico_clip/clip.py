"""
CLIP Module
==================

This module provides extracts features from ExtractedObjectsIU using CLIP.
"""

import clip
from collections import deque
import numpy as np
import threading
from PIL import Image
import time
from retico_core.abstract import IncrementalQueue
import torch

import retico_core
# TODO make is so that you don't need these 3 lines below
# idealy retico-vision would be in the env so you could 
# import it by just using:
# from retico_vision.vision import ImageIU, ExtractedObjectsIU
import sys
# prefix = '../../'
# sys.path.append(prefix+'retico-vision')

from retico_vision.vision import ExtractedObjectsIU, ObjectFeaturesIU

class ClipObjectFeatures(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "CLIP Object Features"

    @staticmethod
    def description():
        return "Module for extracting visual features from images."

    @staticmethod
    def input_ius():
        return [ExtractedObjectsIU]

    @staticmethod
    def output_iu():
        return ObjectFeaturesIU
    
    def __init__(self, model_name = "ViT-B/32", show=False, top_objects=1, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        self.show = show
        self.top_objects = top_objects
        self.queue = deque(maxlen=1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            else:
                self.queue.append(iu)

    # def get_clip_subimage(self, I, img_box):
    #     # expected format:
    #     # Numpy array, length 4, [xmin, ymin, xmax, ymax]

    #     xmin = int(img_box[0])
    #     xmax = int(img_box[2])
    #     ymin = int(img_box[1])
    #     ymax = int(img_box[3])
    #     sub = I.crop([xmin,ymin,xmax,ymax])

    #     if self.show:
    #         import cv2
    #         img_to_show = np.asarray(sub)
    #         cv2.imshow('image',cv2.cvtColor(img_to_show, cv2.COLOR_RGB2BGR)) 
    #         cv2.waitKey(1)
    #     # pim = PImage.fromarray(sub)
    #     sub.load()
    #     return sub 
    
    def _extractor_thread(self):
        while self._extractor_thread_active:
            if len(self.queue) == 0:
                time.sleep(0.5)
                continue

            input_iu = self.queue.popleft()
            image = input_iu.image
            detected_objects = input_iu.extracted_objects
            object_features = {}

            for i, obj in enumerate(detected_objects):
                # sub_img = self.get_clip_subimage(image, obj)
                if i>=self.top_objects: break
                sub_img = detected_objects[obj]
                
                # if self.show:
                #     import cv2
                #     img_to_show = np.asarray(sub_img)
                #     cv2.imshow('image',cv2.cvtColor(img_to_show, cv2.COLOR_RGB2BGR)) 
                #     cv2.waitKey(1)

                if self.show:
                    # print(sub.getbands())
                    # sub = sub.convert("BGR")
                    sub.show()                

                # sub_img = Image.fromarray(sub_img)
                # sub_img.load()

                with torch.no_grad():
                    img = self.preprocess(sub_img).unsqueeze(0).to(self.device)
                    yhat = self.model.encode_image(img).cpu().numpy()
                    object_features[i] = yhat.tolist()

            output_iu = self.create_iu(input_iu)
            output_iu.set_object_features(image, object_features)
            um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
            self.append(um)
    

    def prepare_run(self):
        self.model, self.preprocess = clip.load(self.model_name, self.device)
        self._extractor_thread_active = True
        threading.Thread(target=self._extractor_thread).start()
    
    def shutdown(self):
        self._extractor_thread_active = False