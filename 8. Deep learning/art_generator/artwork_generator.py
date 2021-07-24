import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
import pprint
import matplotlib.pyplot as plt

# 1 - Problem Statement
# Neural Style Transfer (NST) is one of the most fun techniques in deep learning. As seen below, it merges two images,
# namely: a "content" image (C) and a "style" image (S), to create a "generated" image (G).

# 2 - Transfer Learning
# Neural Style Transfer (NST) uses a previously trained convolutional network, and builds on top of that. The idea of
# using a network trained on a different task and applying it to a new task is called transfer learning.
#
# Following the original NST paper, we will use the VGG network. Specifically, we'll use VGG-19, a 19-layer version
# of the VGG network. This model has already been trained on the very large ImageNet database, and thus has learned to
# recognize a variety of low level features (at the shallower layers) and high level features (at the deeper layers).

pp = pprint.PrettyPrinter(indent=1)
model = load_vgg_model("./pretrained-model/imagenet-vgg-verydeep-19.mat")
pp.pprint(model)

# The model is stored in a python dictionary.
# The python dictionary contains key-value pairs for each layer.
# The 'key' is the variable name and the 'value' is a tensor for that layer.

# 3 - Neural Style Transfer (NST)
# We will build the Neural Style Transfer (NST) algorithm in three steps:
#
# Build the content cost function J(C,G)
# Build the style cost function J(S,G)
# Put it together to get J(G) = alpha * J(C,G) + beta * J(S,G).

# 3.1 - Computing the content cost
# In our running example, the content image C will be the picture of the Louvre Museum in Paris. Run the code below to
# see a picture of the Louvre.
import imageio

content_image = imageio.imread("images/louvre.jpg")
imshow(content_image)

# 3.1.1 - Make generated image G match the content of image C

# Shallower versus deeper layers
# The shallower layers of a ConvNet tend to detect lower-level features such as edges and simple textures.
# The deeper layers tend to detect higher-level features such as more complex textures as well as object classes.

def compute_content_cost(a_C, a_G):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.reshape(a_C, shape=[m, n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[m, -1, n_C])

    # compute the cost with tensorflow (≈1 line)
    J_content = 1 / (4 * n_H * n_W * n_C) * tf.reduce_sum(tf.square(tf.subtract(a_C, a_G)))
    return J_content

tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_content = compute_content_cost(a_C, a_G)
    print("J_content = " + str(J_content.eval()))

# Style image
style_image = imageio.imread("images/claude-monet.jpg")
imshow(style_image)


# GRADED FUNCTION: gram_matrix
def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """

    ### START CODE HERE ### (≈1 line)
    GA = tf.matmul(A, tf.transpose(A))
    ### END CODE HERE ###

    return GA


tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    A = tf.random_normal([3, 2 * 1], mean=1, stddev=4)
    GA = gram_matrix(A)

    print("GA = \n" + str(GA.eval()))


# GRADED FUNCTION: compute_layer_style_cost

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    ### START CODE HERE ###
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.transpose(tf.reshape(a_S, shape=[n_H * n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, shape=[n_H * n_W, n_C]))

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (≈1 line)
    J_style_layer = 1 / (4 * (n_C ** 2) * (n_H * n_W) ** 2) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))
    return J_style_layer


tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_style_layer = compute_layer_style_cost(a_S, a_G)

    print("J_style_layer = " + str(J_style_layer.eval()))

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name]
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out

        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """

    ### START CODE HERE ### (≈1 line)
    J = alpha * J_content + beta * J_style
    ### END CODE HERE ###

    return J


tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(3)
    J_content = np.random.randn()
    J_style = np.random.randn()
    J = total_cost(J_content, J_style)
    print("J = " + str(J))


# Reset the graph
tf.reset_default_graph()

# Start interactive session
sess = tf.InteractiveSession()

# content_image = imageio.imread("images/w_hotel.jpg")
# content_image = reshape_and_normalize_image(content_image)
