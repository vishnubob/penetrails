#!/usr/bin/env python
import mxnet as mx
import numpy as np
import os
import shutil
import mimetypes
import PIL.Image as Image
import json
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

def scan_images(path):
    images = []
    for (root, dirs, files) in os.walk(path):
        for fn in files:
            (stem, ext) = os.path.splitext(fn)
            ext = ext.lower()
            ftype = mimetypes.types_map.get(ext, "na")
            if not ftype.startswith("image/"):
                continue
            imgfn = os.path.join(root, fn)
            images.append(imgfn)
    return images

def load_image(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img

def checkpoint(data, fn="class.json"):
    if os.path.exists(fn):
        os.rename(fn, fn + ".bak")
    with open(fn, 'w') as fh:
        json.dump(data, fh)

def run():
    mimetypes.init()
    path='http://data.mxnet.io/models/imagenet-11k/'
    mx.test_utils.download(path+'resnet-152/resnet-152-symbol.json')
    mx.test_utils.download(path+'resnet-152/resnet-152-0000.params')
    mx.test_utils.download(path+'synset.txt')
    sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-152', 0)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))], label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)

    with open('synset.txt', 'r') as f:
        labels = [l.rstrip() for l in f]

    images = scan_images("./images")
    report = {}
    for (idx, path) in enumerate(images):
        print(path)
        report[path] = []
        try:
            img = load_image(path)
        except KeyboardInterrupt:
            raise
        except:
            continue
        batch = Batch([mx.nd.array(img)])
        mod.forward(batch)
        prob = mod.get_outputs()[0].asnumpy()
        prob = np.squeeze(prob)
        a = np.argsort(prob)[::-1]
        for (idx, i_idx) in enumerate(a[0:5]):
            row = dict(probability=float(prob[i_idx]), class_id=labels[i_idx])
            if idx == 0:
                print(row)
            report[path].append(row)
        if ((idx + 1) % 2) == 0:
            checkpoint(report)

if __name__ == "__main__":
    run()
