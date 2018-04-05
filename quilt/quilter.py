#!/usr/bin/env python
#
# CNN-GAN built for inventing quilt designs
# Giles Hall, April 2018
#
# Script is based on MNIST hand writing GAN by Sherlock Liao
# On GitHub:
#   SherlockLiao/mxnet-gluon-tutorial/09-Generative Adversarial network/conv_gan.py
#
# I consider myself a novice in the field of applied neural networks, so please
# take my code with a grain of salt

import operator
import bisect
from PIL import Image
import os
import time
from threading import Lock, Thread, Condition
from queue import Queue

import numpy as np
import mxnet as mx
from mxnet import gluon as g
from mxnet import ndarray as nd
import sys
import requests

def gcd(a,b):
    while b > 0:
        a, b = b, a % b
    return a
    
def find_res(cnt, rev=True, ratio=.5):
    vals = set([gcd(cnt, x) for x in range(1, cnt)])
    rats = []
    for width in vals:
        height = cnt / width
        rat = width / height
        rats.append((rat, width, height))
    rats = sorted(rats, key=operator.itemgetter(0))
    keys = [rat[0] for rat in rats]
    idx = bisect.bisect_left(keys, ratio)
    idx = min(len(rats) - 1, idx)
    width = max(rats[idx][1:])
    height = min(rats[idx][1:])
    return list(map(int, (width, height)))

def norm_ip(img, min, max):
    img = np.clip(img, min, max)
    img = (img - min) / (max - min + 1e-5)
    return img

def norm_range(t, range=None):
    if range is not None:
        return norm_ip(t, range[0], range[1])
    else:
        return norm_ip(t, t.min(), t.max())

def tile_image(imgs):
    imgcnt = imgs.shape[0]
    (prow, pcol) = find_res(imgcnt)
    tiled = []
    total = prow * pcol
    for batch in range(0, prow):
        up = pcol * batch
        down = pcol * (batch + 1)
        subs = np.concatenate(imgs[up:down], 1)
        tiled.append(subs)
    return np.concatenate(tiled, 2)

def save_image(data, filename):
    im = data.asnumpy()
    im = 0.5 * (im + 0.5)
    im = (np.clip(im, 0, 1) * 255.0).astype(np.uint8)
    im = tile_image(im)
    im = Image.fromarray(im.T, mode='RGB')
    im.save(filename)

def img_transform(data):
    data = data.reshape((64, 64, 3)).astype((np.uint8))
    data = (data.astype('float32') / 255 - 0.5) / 0.5
    return data.T

batch_size = 16
num_epoch = 5000
z_dimension = 100


# Discriminator
class discriminator(g.HybridBlock):
    def __init__(self):
        super(discriminator, self).__init__()
        with self.name_scope():
            self.conv1 = g.nn.HybridSequential(prefix='conv1_')
            with self.conv1.name_scope():
                self.conv1.add(g.nn.Conv2D(1024, (3, 3), in_channels=3, padding=2))
                self.conv1.add(g.nn.LeakyReLU(0.2))
                self.conv1.add(g.nn.AvgPool2D((2, 2)))

            self.conv2 = g.nn.HybridSequential(prefix='conv2_')
            with self.conv2.name_scope():
                self.conv2.add(g.nn.Conv2D(512, 3, padding=2))
                self.conv2.add(g.nn.LeakyReLU(0.2))
                self.conv2.add(g.nn.AvgPool2D((2, 2)))

            self.fc = g.nn.HybridSequential(prefix='fc_')
            with self.fc.name_scope():
                self.fc.add(g.nn.Flatten())
                self.fc.add(g.nn.Dense(1024))
                self.fc.add(g.nn.LeakyReLU(0.2))
                self.fc.add(g.nn.Dense(1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x

# Generator
class generator(g.HybridBlock):
    def __init__(self, num_feature):
        super(generator, self).__init__()
        with self.name_scope():
            self.fc = g.nn.Dense(num_feature, use_bias=False)

            self.br = g.nn.HybridSequential(prefix='batch_relu_')
            with self.br.name_scope():
                self.br.add(g.nn.BatchNorm())
                self.br.add(g.nn.Activation('relu'))

            self.downsample = g.nn.HybridSequential(prefix='ds_')
            with self.downsample.name_scope():
                self.downsample.add(g.nn.Conv2D(50, 3, strides=1, padding=1))
                self.downsample.add(g.nn.BatchNorm())
                self.downsample.add(g.nn.Activation('relu'))
                self.downsample.add(g.nn.Conv2D(25, 3, strides=1, padding=1))
                self.downsample.add(g.nn.BatchNorm())
                self.downsample.add(g.nn.Activation('relu'))
                self.downsample.add(g.nn.Conv2D(3, 2, strides=2, activation='tanh'))

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape((x.shape[0], 3, 128, 128))
        x = self.br(x)
        x = self.downsample(x)
        return x

#ge = generator(3136)

def _get_dataset(cnt):
    now = time.time()
    # set this to your tile server host
    url = "http://localhost:10999/get/dataset/%d" % cnt
    while True:
        r = requests.get(url)
        if r.status_code != 200:
            time.sleep(5)
            continue
        break
    data = r.json()
    data = np.array(data)
    data = mx.gluon.data.SimpleDataset(data)
    data = data.transform(img_transform)
    lapse = time.time() - now
    print("request: got %d images in %s seconds" % (len(data), lapse))
    return mx.gluon.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True)

def dataset_thread(cnt):
    global ds_cv, dataset, que
    que = Queue(5)
    next_dataset = None
    ds_cv = Condition()
    while True:
        ds = _get_dataset(cnt)
        ds_cv.acquire()
        while que.full():
            ds_cv.wait()
        que.put(ds)
        ds_cv.notify()
        ds_cv.release()

last_ds = None
def get_dataset():
    global ds_cv, que, last_ds
    ds_cv.acquire()
    if que.empty():
        if last_ds is None:
            while que.empty():
                ds_cv.wait()
            last_ds = _ds = que.get()
            ds_cv.notify()
        else:
            _ds = last_ds
    else:
        last_ds = _ds = que.get()
        ds_cv.notify()
    ds_cv.release()
    return _ds

def init_gan(ctx, params=None):
    d = discriminator()
    d.hybridize()
    ge = generator(3 * 128 * 128)
    ge.hybridize()
    if params is None:
        d.collect_params().initialize(mx.init.Xavier(), ctx)
        ge.collect_params().initialize(mx.init.Xavier(), ctx)
    else:
        (d_data, ge_data) = params
        d.load_params(d_data, ctx)
        ge.load_params(ge_data, ctx)
    return (d, ge)

def train_gan(ctx, net=None, initial_epoch=0):
    t = Thread(target=dataset_thread, args=(16 * 100 * 2,))
    t.start()

    # Binary cross entropy loss and optimizer
    bce = g.loss.SigmoidBinaryCrossEntropyLoss()

    if net is None:
        net = init_gan(ctx)
    (d, ge) = net
    d_optimizer = g.Trainer(d.collect_params(), 'adam', {'learning_rate': 0.0003})
    g_optimizer = g.Trainer(ge.collect_params(), 'adam', {'learning_rate': 0.0003})

    # Start training
    epoch = initial_epoch
    while True:
        epoch += 1
        dataloader = get_dataset()
        i = 0
        for rep in range(5):
            i_rep = i
            for i, img in enumerate(dataloader):
                num_img = img.shape[0]
                # =================train discriminator
                real_img = img.as_in_context(ctx)
                real_label = nd.ones(shape=[num_img], ctx=ctx)
                fake_label = nd.zeros(shape=[num_img], ctx=ctx)

                # compute loss of real_img
                with mx.autograd.record():
                    real_out = d(real_img)
                    d_loss_real = bce(real_out, real_label)
                real_scores = real_out  # closer to 1 means better

                # compute loss of fake_img
                z = nd.random_normal(
                    loc=0, scale=1, shape=[num_img, z_dimension], ctx=ctx)
                with mx.autograd.record():
                    fake_img = ge(z)
                    fake_out = d(fake_img)
                    d_loss_fake = bce(fake_out, fake_label)
                fake_scores = fake_out  # closer to 0 means better

                # bp and optimize
                with mx.autograd.record():
                    d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step(num_img)

                # ===============train generator
                # compute loss of fake_img
                with mx.autograd.record():
                    fake_img = ge(z)
                    output = d(fake_img)
                    g_loss = bce(output, real_label)

                # bp and optimize
                g_loss.backward()
                g_optimizer.step(num_img)

                #save_image(real_img, './img/real_images.png')
                if (i + i_rep + 1) % 100 == 0:
                    print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '
                          'D real: {:.6f}, D fake: {:.6f}'.format(
                              epoch, num_epoch,
                              nd.mean(d_loss).asscalar(),
                              nd.mean(g_loss).asscalar(),
                              nd.mean(real_scores).asscalar(),
                              nd.mean(fake_scores).asscalar()))
        if ((epoch + 1) % 5) == 0:
            save_image(real_img, './img/real_images.png')
            save_image(fake_img, './img/fake_images-{}.png'.format(epoch + 1))
            d.save_params('./dis-quilt-{}.params'.format(epoch + 1))
            ge.save_params('./gen-quilt-{}.params'.format(epoch + 1))

if __name__ == "__main__":
    params = None
    ctx = mx.gpu()
    if len(sys.argv) > 1:
        epoch = int(sys.argv[1])
        d_fn = sys.argv[2]
        ge_fn = sys.argv[3]
        params = (d_fn, ge_fn)
    net = init_gan(ctx, params)
    train_gan(ctx, net=net, initial_epoch=epoch)
