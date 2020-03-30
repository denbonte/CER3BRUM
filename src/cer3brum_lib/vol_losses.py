## ----------------------------------------
## Function to optimise during the training
## of volumetric models (input's 3D)
## ----------------------------------------
## 
## ----------------------------------------
## Author: Dennis Bontempi, Michele Svanera
## Version: 2.0
## Email: dennis.bontempi@glasgow.ac.uk
## Status: ready to use
## Modified: 20 Feb 19
## ----------------------------------------

import numpy as np
import tensorflow as tf

from keras import backend as K

## -----------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------

# adapted from:
# https://github.com/keras-team/keras/issues/9395

# the implementation follows
# https://arxiv.org/pdf/1706.05721.pdf

# alpha = beta = 0.5 : dice coefficient
# alpha = beta = 1   : tanimoto coefficient (also known as extended jaccard)
# alpha + beta = 1   : produces set of F*-scores

# the following function is used mainly for testing (different trials with alpha and beta..)
# and for explanatory purposes - the actual loss function follows after this

def tversky_loss(y_true, y_pred):
    
    # tversky coeff alpha and beta
    # if bot are set to be 0.5, the DC is obtained (exactly as from the "dice_coef_multilabel" func)
    alpha = 0.5
    beta  = 0.5
    
    # in our case, initialize a tensor of shape NxMxPxNCLASSES (M, N and P are the input dim.s)
    # NOTE: "y_true" is a NxMxPxNCLASSES tensor that contains the information about the correct
    # segmentation class (one-hot-encoded, 1.0 in that class and 0.0 in all the others)
    ones = K.ones(K.shape(y_true))
    
    # following the paper notation, name "p0" the classes prediction tensor (probability)
    p0 = y_pred
    
    # therefore, 1 - p0 is the probability that the particular voxel DOES NOT belong to that class 
    p1 = ones - y_pred
    
    # the "y_true" tensor contains 1.0 only in the channel associated to the correct class 
    # (and 0.0 in all the others) (see the example below) 
    g0 = y_true
    
    # .. so that the following quantity is a mask that can be used to keep only the probability
    # that a wrong guess is made (since the only one zeroed will be the correct one) 
    g1 = ones - y_true
    
    # .. for instance, p0[10, 10, 10, :] gives us a TF tensor (convertible to numpy array..)
    # containing the probability that the voxel at [10, 10, 10] belongs to each class
    #
    # e.g. (in case of 8 classes, GT = background):
    #       y_true[10, 10, 10, :] = g0[10, 10, 10, :] = [1.0, .0, .0, .0, .0, .0, .0, .0]
    #       y_pred[10, 10, 10, :] = p0[10, 10, 10, :] = [0.6, .1, .0, .1, .2, .0, .0, .0]
    
    # NOTE: when dealing with TF tensors, "*" in interpreted as element-wise multiplication 
    # ... so the giant sum at the numerator can be computed by element-wise multiplying the two 
    # tensors and then by computing a sum on every axis
    num = K.sum(p0*g0, (0,1,2,3))
    
    # the denominator is composed by three terms:
    #   - the numerator, so that since we're dealing with probabilities and the other two terms
    #       will always be positive, the tversky coefficient is always bounded between 0 and 1;
    #   - the tensor p0*g1 (predicted probabilities masked with g0) contains for each voxel the 
    #       probability that the classification is wrong (false positive);
    #   - the tensor p1*g0 (probability each voxel does not belong to a class masked with the
    #       tensor containing one only in the correct class) contains the probability that the 
    #       correct class is not identified (false negative);
    
    # so about the tversky coefficient:
    #   - if both the second and the third term at the denominator are small (meaning the 
    #       segmentation is quite good), then the tversky coefficient is close to one;
    #   - as alpha increases, the importance of the false positives increases;
    #   - as beta increases, the importance of the false negatives increases;
    den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))
    
    # compute the sum of the resulting tensor (should have dimension 1xNCLASSES)
    # this will be a number between 0 and num_classes
    ratio = K.sum(num/den) 
    
    # get the number of the classes as a tensor
    num_classes = K.cast(K.shape(y_true)[-1], 'float32')
    
    # compute the difference between the two (this will lead to something similar to the 
    # "dice_coef_multilabel", that is the sum of all the DCs associated to each class)
    return num_classes - ratio

## -----------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------

def tanimoto_coefficient(y_true, y_pred):
    
    # starting from the Tversky index and setting both alpha and beta equal to 1 we can obtain the
    # Tanimoto Coefficient (extended Jaccard Coefficient)
    # for any additional information about the quantities, see the "tversky_loss" function
    alpha = 1.0
    beta  = 1.0

    ones = K.ones(K.shape(y_true))
    
    p0 = y_pred
    p1 = ones - y_pred
    g0 = y_true
    g1 = ones - y_true
    
    eps = 1e-7
    
    num = K.sum(p0*g0, (0,1,2,3)) + eps
    den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))
    
    ratio = K.sum(num/den) 
    
    num_classes = K.cast(K.shape(y_true)[-1], 'float32')
    
    return num_classes - ratio

## -----------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------


def dice_coef_multilabel(y_true, y_pred):
    
    # starting from the Tversky index and setting both alpha and beta equal to .5 we can obtain the
    # Dice Coefficient.
    # for any additional information about the quantities, see the "tversky_loss" function
    alpha = 0.5
    beta  = 0.5

    ones = K.ones(K.shape(y_true))
    
    p0 = y_pred
    p1 = ones - y_pred
    g0 = y_true
    g1 = ones - y_true
    
    eps = 1e-7
    
    num = K.sum(p0*g0, (0,1,2,3)) + eps
    den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))
    
    ratio = K.sum(num/den) 
    
    num_classes = K.cast(K.shape(y_true)[-1], 'float32')
    
    return num_classes - ratio

## -----------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------

def log_dice_coef_multilabel(y_true, y_pred):
    
    # starting from the Tversky index and setting both alpha and beta equal to .5 we can obtain the
    # Dice Coefficient.
    # for any additional information about the quantities, see the "tversky_loss" function
    alpha = 0.5
    beta  = 0.5

    ones = K.ones(K.shape(y_true))
    
    p0 = y_pred
    p1 = ones - y_pred
    g0 = y_true
    g1 = ones - y_true
    
    eps = 1e-7
    
    num = K.sum(p0*g0, (0,1,2,3)) + eps
    den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))
    
    ratio = K.sum(num/den) 
    
    num_classes = K.cast(K.shape(y_true)[-1], 'float32')
    
    # As suggested in several reports
    # (e.g. https://ai.intel.com/biomedical-image-segmentation-u-net/#gs.LbZTBfi4)
    # instead of minimize the quantity (num_classes - ratio), e.g. (8 - multiclass_dice),
    # one could try to minimize directly -log of the dice.
    # In our case, since the DC is not limited between 0 and 1, we need to normalize the quantity
    # obrained from the sum by the number of classes beforehand
    
    return -tf.log(ratio/num_classes)

## -----------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------

"""
# another way to compute the dice coefficient is to define a function that takes care of the 2D
# computation (can be used as loss in the 2D models too...) and then define another function that
# simply computes the DC for every class and weight them (in potentially different ways).

# The only problem with this approach is that due to the way in which TF treats loss functions
# it is not possible to use at run-time the number of classes to initialize a "range()" useful
# to sum on all the classes   

"""

# https://github.com/keras-team/keras/issues/9395

# DC(A, B) = (2*A intersection B)/(#A + #B) 

# KEEP IN MIND: since the backend is Tensorflow, then the format of y_true and y_pred 
# (that are numpy arrays converted in TF tensors) will be:

# (BATCH_SIZE, DIM_1, DIM_2, ... DIM_N, NUM_CLASSES)

# single class dice coefficient
def dice_coef(y_true, y_pred):
    
    # the smooth coefficient act as regularizer: its value altough depend from the quantity at 
    # stake (e.g. values of K.sum() ... )
    eps = 1e-7
    
    # "unwrap" the ground truth and the predicted volumes to obtain a 1xN tensors 
    # (DC doesn't take into account spatial information) (boolean vector)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    #y_pred_f = K.flatten(tf.argmax(y_pred, 3))
    
    # compute the intersection between the two sets as the product between the unwrapped boolean 
    # tensors computed above (use K.sum since the quantity is still a tensor)
    intersection = K.sum(y_true_f * y_pred_f)
    
    # apply the definition of DC
    return (2. * intersection + eps) / (K.sum(y_true_f) + K.sum(y_pred_f) + eps)

## -----------------------------------------------------------------------------------------------

"""
# to compute the dice for a multilabel problem, just sum all the dice coefficients
def dice_coef_multilabel(y_true, y_pred):
    
    #FIXME: doesn't work yet --> https://github.com/keras-team/keras/issues/4075
    
    # it's impossible at this stage to compute the shape of a tensor, since Keras/TF use this 
    # function to generate and compile a C function that is then called in training-time
    
    # get the number of classes exploiting the last dimension of the y_true tensor
    # NOTE: K.shape(y_true)[-1] returns the value we're looking for, but it still a tensor
    # (.. like almost every function does in tf.keras.backend).
    # So use K.eval() to obtain the plain (numpy float) value (int32)
    #num_labels = K.eval(K.cast(K.shape(y_true)[-1], 'float32')) 
    num_labels = K.cast(K.shape(y_true)[-1], 'float32')
    num_labels = 8
    
    # initialize the multiclass dice loss coefficient as "num_labels"
    multiclass_dice = num_labels
    
    # total multiclass dice coefficient computation
    for label in range(num_labels):
        multiclass_dice -= dice_coef(y_true[:, :, :, :, label], y_pred[:, :, :, :, label])
        
    return multiclass_dice
"""

## -----------------------------------------------------------------------------------------------

# to compute the dice for a multilabel problem, just sum all the dice coefficients
def average_dice_coef_multilabel(y_true, y_pred):
    
    #FIXME: doesn't work yet --> https://github.com/keras-team/keras/issues/4075
    
    # it's impossible at this stage to compute the shape of a tensor, since Keras/TF use this 
    # function to generate and compile a C function that is then called in training-time
    
    # get the number of classes exploiting the last dimension of the y_true tensor
    # NOTE: K.shape(y_true)[-1] returns the value we're looking for, but it still a tensor
    # (.. like almost every function does in tf.keras.backend).
    # So use K.eval() to obtain the plain (numpy float) value (int32)
    #num_labels = K.eval(K.shape(y_true)[-1])
    num_labels = K.cast(K.shape(y_true)[-1], 'float32')
    num_labels = 8
    
    # initialize the multiclass dice loss coefficient as "num_labels"
    multiclass_dice = float(num_labels)
    
    # average multiclass dice coefficient computation
    for label in range(num_labels):
        multiclass_dice -= dice_coef(y_true[:, :, :, :, label], y_pred[:, :, :, :, label])
        
    return multiclass_dice/float(num_labels)

    
## -----------------------------------------------------------------------------------------------

# to compute the dice for a multilabel problem, just sum all the dice coefficients
def weighted_dice_coef_multilabel(y_true, y_pred):

    #FIXME: doesn't work yet --> https://github.com/keras-team/keras/issues/4075  
    
    # it's impossible at this stage to compute the shape of a tensor, since Keras/TF use this 
    # function to generate and compile a C function that is then called in training-time
    
    # get the number of classes exploiting the last dimension of the y_true tensor
    # NOTE: K.shape(y_true)[-1] returns the value we're looking for, but it still a tensor
    # (.. like almost every function does in tf.keras.backend).
    # So use K.eval() to obtain the plain (numpy float) value (int32)
    #num_labels = K.eval(K.shape(y_true)[-1])
    num_labels = 8
        
    # weights vector obtained from the dataset analysis, since all that follows is symbolical and
    # it's impossible at this stage to compute the number of voxels that belongs to a class,
    # say exploiting y_true (Keras/TF use this function to generate and compile a C function that
    # is then called in training-time...)
    
    # the idea is: define weights such that the final coefficient (sum of the weighted DC for 
    # each label) will be a number between 0 and 1 and so that it mitigates the unbalanced 
    # training set
    percentage_freq = np.array([84.35, 5.55, 0.75, 4.45, 3.7, 0.5, 0.25, 0.45])
     
    # weighting method taken from: 
    # https://arxiv.org/pdf/1801.05912.pdf
    
    # quadratic weighting:
    
    N   = 250
    eps = 1e-7 
    
    label_weights   = N/(num_labels * percentage_freq * percentage_freq + eps)  

    label_weights = label_weights/np.sum(label_weights)

    # initialize the multiclass dice loss coefficient as "num_labels"
    multiclass_dice = num_labels
    
    # weighted multiclass dice coefficient computation
    for label in range(num_labels):
        multiclass_dice -= label_weights[label] * dice_coef(y_true[:, :, :, :, label], y_pred[:, :, :, :, label])
        
    return multiclass_dice


## -----------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------    

losses_dict = {'categorical_crossentropy'       : 'categorical_crossentropy',
               'multiclass_dice_coeff'          : dice_coef_multilabel,
               'average_multiclass_dice_coeff'  : average_dice_coef_multilabel,
               'weighted_multiclass_dice_coeff' : weighted_dice_coef_multilabel,
               'tversky_loss'                   : tversky_loss,
               'tanimoto_coefficient'           : tanimoto_coefficient,
               'log_multiclass_dice_coeff'      : log_dice_coef_multilabel}
    
    
    
    

