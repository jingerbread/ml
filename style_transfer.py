from __future__ import print_function

import time
from PIL import Image
import numpy as np

from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

height = 512
width = 512

content_image_path = 'images/hugo.png'
content_image = Image.open(content_image_path)
content_image = content_image.resize((width, height))
content_image.show()

style_image_path = 'images/styles/wave.png'
style_image = Image.open(style_image_path)
style_image = style_image.resize((width, height))
style_image.show()

content_array = np.asarray(content_image, dtype='float32')
content_array = np.expand_dims(content_array, axis=0)
print(content_array.shape)

style_array = np.asarray(style_image, dtype='float32')
style_array = np.expand_dims(style_array, axis=0)
print(style_array.shape)

# Then, we convert these images into a form suitable for numerical processing.
# In particular, we add another dimension (beyond the classic height x width x 3 dimensions)
# so that we can later concatenate the representations of these two images into a common data structure.
content_array = np.asarray(content_image, dtype='float32')
content_array = np.expand_dims(content_array, axis=0)
print(content_array.shape)

style_array = np.asarray(style_image, dtype='float32')
style_array = np.expand_dims(style_array, axis=0)
print(style_array.shape)

# Subtract the mean RGB value (computed previously on the ImageNet training set
# and easily obtainable from Google searches) from each pixel.
# Flip the ordering of the multi-dimensional array from RGB to BGR (the ordering used in the paper).
content_array[:, :, :, 0] -= 103.939
content_array[:, :, :, 1] -= 116.779
content_array[:, :, :, 2] -= 123.68
content_array = content_array[:, :, :, ::-1]

style_array[:, :, :, 0] -= 103.939
style_array[:, :, :, 1] -= 116.779
style_array[:, :, :, 2] -= 123.68
style_array = style_array[:, :, :, ::-1]

# Now we’re ready to use these arrays to define variables in Keras’ backend (the TensorFlow graph).
# We also introduce a placeholder variable to store the combination image
# that retains the content of the content image while incorporating the style of the style image.
content_image = backend.variable(content_array)
style_image = backend.variable(style_array)
combination_image = backend.placeholder((1, height, width, 3))

# Finally, we concatenate all this image data into a single tensor that’s suitable for processing by Keras’ VGG16 model.
input_tensor = backend.concatenate([content_image, style_image, combination_image], axis=0)

# Note that by setting include_top=False in the code below, we don’t include any of the fully-connected layers.
model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
# Let’s make a list of these names so that we can easily refer to individual layers later.
layers = dict([(layer.name, layer.output) for layer in model.layers])
layers

# These are arbitrary, but the following set have been chosen after quite a bit of experimentation
# to find a set that generates output that’s aesthetically pleasing to me.
content_weight = 0.025
style_weight = 5.0
total_variation_weight = 1.0

# We’ll now use the feature spaces provided by specific layers of our model to define these three loss functions.
# We begin by initialising the total loss to 0 and adding to it in stages.
loss = backend.variable(0.)
# The content loss is the (scaled, squared) Euclidean distance between feature representations
#  of the content and combination images.
def content_loss(content, combination):
    return backend.sum(backend.square(combination - content))

layer_features = layers['block2_conv2']
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

loss += content_weight * content_loss(content_image_features, combination_features)

# For the style loss, we first define something called a Gram matrix.
# The Gram matrix can be computed efficiently by reshaping the feature spaces suitably and taking an outer product.
def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram


# The style loss is then the (scaled, squared)
# Frobenius norm of the difference between the Gram matrices of the style and combination images.
def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = height * width
    return backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


feature_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
for layer_name in feature_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl


# If you were to solve the optimisation problem with only the two loss terms we’ve introduced so far
# (style and content), you’ll find that the output is quite noisy. We thus add another term,
# called the total variation loss (a regularisation term) that encourages spatial smoothness.
def total_variation_loss(x):
    a = backend.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = backend.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))


loss += total_variation_weight * total_variation_loss(combination_image)

# Define needed gradients and solve the optimisation problem
grads = backend.gradients(loss, combination_image)

# We then introduce an Evaluator class that computes loss and gradients in one pass
# while retrieving them via two separate functions, loss and grads.
# This is done because scipy.optimize requires separate functions for loss and gradients,
# but computing them separately would be inefficient.
outputs = [loss]
outputs += grads
f_outputs = backend.function([combination_image], outputs)


def eval_loss_and_grads(x):
    x = x.reshape((1, height, width, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()

# We stop after 10 iterations because the output looks good to me and the loss stops reducing significantly.
x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

iterations = 10

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))

# Note that we need to subject our output image to the inverse of the transformation
# we did to our input images before it makes sense.
x = x.reshape((height, width, 3))
x = x[:, :, ::-1]
x[:, :, 0] += 103.939
x[:, :, 1] += 116.779
x[:, :, 2] += 123.68
x = np.clip(x, 0, 255).astype('uint8')

result_image = Image.fromarray(x)
result_image.show()