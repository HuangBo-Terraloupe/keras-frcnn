from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils

from keras_frcnn import resnet as nn

input_shape_img = (600, 600, 3)

anchor_box_scales = [128, 256, 512]
# anchor box ratios
anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]

num_rois = 32

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(shared_layers, roi_input, num_rois, nb_classes=21, trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
model_all = Model([img_input, roi_input], rpn[:2] + classifier)


optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier,
                         loss=[losses.class_loss_cls, losses.class_loss_regr(21-1)],
                         metrics={'dense_class_{}'.format(21): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

print model_all.summary()