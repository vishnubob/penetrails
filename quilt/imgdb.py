#!/usr/bin/env python

import time
import struct
import os
import shutil
import argparse
import mimetypes
import random
import numpy as np
import pickle
from multiprocessing import Process, Queue
from imgaug import augmenters as iaa
import threading

from PIL import Image
from datetime import datetime, timedelta
from functools import lru_cache

mimetypes.init()
mimetypes.add_type("image/x-canon-cr2", ".cr2")

@lru_cache(maxsize=32)
def get_image_cached(path):
    img = Image.open(path)
    img = img.convert("RGB")
    img = np.array(img).astype(np.uint8)
    return img

class ImageRow(object):
    def __init__(self, imgfn):
        self.path = os.path.abspath(imgfn)
        (stem, ext) = os.path.splitext(imgfn)
        self.ext = ext.lower()
        self._pil_image = None
        if ext == ".jpeg":
            ext = ".jpg"

    @property
    def is_ok(self):
        try:
            size = self.image.size
        except:
            return False
        return True

    @property
    def image(self):
        return get_image_cached(self.path)

    def asdict(self):
        return {'ext': self.ext, 'path': self.path, 'image': self}

    def __cmp__(self, other):
        return cmp(self.timestamp, other.timestamp)

class Proc(Process):
    MaxSize = None

    def __init__(self, tt_period=5, que=None, log=True, **kw):
        super(Proc, self).__init__()
        self.daemon = True
        self.que = que if que is not None else Queue(self.MaxSize)
        self.running = True
        self.tick = time.time()
        self.tt_period = tt_period
        self.log_flag = log

    def log(self, msg):
        if not self.log_flag:
            return
        msg = "%s: %s" % (self.__class__.__name__, msg)
        print(msg)

    def ticktock(self):
        now = time.time()
        if (now - self.tick) > self.tt_period:
            self.alarm()
            self.tick = now

    def alarm(self):
        qsize = self.que.qsize()
        msg = "qsize=%d, proc=%d" % (qsize, self.n_process)
        self.log(msg)

    def run(self):
        self.log("starting")
        self.n_process = 0
        while self.running:
            self.ticktock()
            self.n_process += self.process()

class BatchScout(Proc):
    MaxSize = 5000

    def __init__(self, loader, size=(64, 64), batch_size=16, shots_on_goal=5, **kw):
        super(BatchScout, self).__init__(**kw)
        self.loader = loader
        self.size = size
        self.batch_size = batch_size
        self.shots_on_goal = 5
        self.mmp = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Sometimes(0.8,
                iaa.Affine(
                    scale=(0.8, 1.5),
                    rotate=(-5, 5),
                    shear=(-5, 5),
                )
            ),
            iaa.Crop(percent=(0, 0.2)),
            iaa.Sometimes(0.1,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),
            iaa.Add((0.8, 2), per_channel=0.5),
            iaa.Scale({"width": self.size[0], "height": self.size[1]})
        ])

    def process(self):
        images = self.loader.get_images(self.batch_size)
        for cnt in range(self.shots_on_goal):
            batch = self.make_me_pretty(images)
            for img in batch:
                self.que.put(img)
        return (self.shots_on_goal * len(images))

    def make_me_pretty(self, images):
        return self.mmp.augment_images(images)

    def get_image(self):
        return self.que.get()

class ImageLoader(Proc):
    MaxSize = 200

    def __init__(self, import_dir='.', **kw):
        super(ImageLoader, self).__init__(**kw)
        self.import_dir = import_dir
        self.scan_images()

    def _load_one_image(self):
        while True:
            idx = random.randint(0, len(self.images) - 1)
            img = self.images[idx]
            if not img.is_ok:
                msg = "removing bad image: %s" % img.path
                self.log(msg)
                del self.images[idx]
                continue
            return img

    def process(self):
        img = self._load_one_image()
        self.que.put(img.image)
        return 1

    def scan_images(self):
        images = []
        for (root, dirs, files) in os.walk(self.import_dir):
            for fn in files:
                (stem, ext) = os.path.splitext(fn)
                ext = ext.lower()
                ftype = mimetypes.types_map.get(ext, "na")
                if not ftype.startswith("image/"):
                    continue
                imgfn = os.path.join(root, fn)
                img = ImageRow(imgfn)
                images.append(img)
        self.images = images

    def get_images(self, n_images):
        images =  []
        for x in range(n_images):
            img = self.que.get()
            images.append(img)
        return images

def init_imgdb(path=".", n_scouts=1):
    global loader
    global scouts
    loader = ImageLoader(path)
    loader.start()
    #
    que = Queue(BatchScout.MaxSize)
    scouts = []
    for n in range(n_scouts):
        lf = True if n == 0 else False
        scout = BatchScout(loader, que=que, log=lf)
        scout.start()
        scouts.append(scout)
    return (loader, scouts, que)
