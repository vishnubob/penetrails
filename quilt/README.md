# Convolutional Generative Adversarial Network to Synthesize Novel Quilt Designs

This is my first attempt at designing and tuning a GAN network for visual
media.  Quilts as a subject were selected because:

1) There are hundreds of thousands of quilt images available on the internet
2) Most images of quilts take up the full frame
3) The regular patterns of geometrical shapes (hopefully) provide toeholds for feature detectors and replicators
4) Quilts are beautiful

[This script](https://github.com/SherlockLiao/mxnet-gluon-tutorial/blob/master/09-Generative%20Adversarial%20network/conv_gan.py)
served as an initial jumping off point for the design.  The topology of the
original network is mostly the same, but a few adjustments were made to
accommodate  RGB channels and an overall doubling of the resolution.

The image corpus is composed of about 40,000 images of quilts scraped from the
web.  ResNet was used to refine this down to about 11,000 images, with a
high-confidence to their, er, "quilt-e-ness".  All of this is done on two home
servers.  One server runs a custom flask app, serving up variations of the
corpus with [imgaug](https://github.com/aleju/imgaug), while the other trains
the GAN on its GPU.

The [results](https://twitter.com/penetrails) don't really look like quilts,
but there is still a lot of headroom for tuning.  In no specific order:

- Reduce generator dimensionality
- Stronger normalization
- Smaller, tighter exemplar image set
- Widen scope of image set as training progresses
- More layers, because why not
- Instrumenting stats via TensorBoard
- Suggestions?

## Training Example

![training quilts](https://github.com/vishnubob/penetrails/raw/master/quilt/examples/training-example.jpg)

## Late Epoch

![fake quilts, late epoch](https://github.com/vishnubob/penetrails/raw/master/quilt/examples/fake-example-late-epoch.jpg)

## Mid Epoch

![fake quilts, mod epoch](https://github.com/vishnubob/penetrails/raw/master/quilt/examples/fake-example-mid-epoch.jpg)

## Early Epoch

![fake quilts, early epoch](https://github.com/vishnubob/penetrails/raw/master/quilt/examples/fake-example-early-epoch.jpg)


