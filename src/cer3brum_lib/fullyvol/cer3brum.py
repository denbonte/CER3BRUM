
## ----------------------------------------
## Fully-volumetric models definition
## ----------------------------------------
## 
## ----------------------------------------
## Author: Dennis Bontempi, Michele Svanera
## Version: 2.0
## Email: dennis.bontempi@glasgow.ac.uk
## Status: ready to use
## Modified: 20 Feb 19
## ----------------------------------------


# add the parent directory to the python path (import vol_losses)
import sys
sys.path.insert(0, '../')

from keras.models import Model, load_model
from keras.layers import Input, UpSampling3D
from keras.layers.core import Dropout, Lambda, Activation
from keras.layers.merge import Concatenate, Add
from keras.layers.pooling import MaxPooling3D
from keras.layers.convolutional import Conv3D

import keras.optimizers
import tensorflow as tf

from cer3brum_lib.vol_losses import losses_dict

from cer3blocks import myConv3DBlock
from cer3blocks import myDownConv3DBlock
from cer3blocks import myUpConv3DBlock



## -----------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------
##
##                                        FULLY VOLUMETRIC
##
## -----------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------

# Three layers CNN, first pooling has factor 4

def ThreeLevelsMaxPool(input_dims,
                       num_classes,
                       init = 'glorot_normal',
                       encoder_act_function = 'elu',
                       decoder_act_function = 'relu',
                       classification_act_function = 'softmax', 
                       loss_function = 'categorical_crossentropy',
                       learning_rate = 1e-05,
                       min_filters_per_layer = 16,
                       use_kernel_reg = False,
                       use_dropout = False,
                       ):

    
    if use_kernel_reg == True:
        lvl2_reg = l2(0.005)
        lvl3_reg = l2(0.01)
    else:
        lvl2_reg = None
        lvl3_reg = None
        
    if use_dropout == True:
        lvl1_dropout = 0.0
        lvl2_dropout = 0.25
        lvl3_dropout = 0.5
    else:
        lvl1_dropout = 0.0 
        lvl2_dropout = 0.0
        lvl3_dropout = 0.0

    lvl1_filters = (3, 3, 3)
    lvl2_filters = (3, 3, 3)
    lvl3_filters = (3, 3, 3)

    # -----------------------------------
    #         ENCODER - LEVEL 1
    # -----------------------------------

    with tf.device('/gpu:0'):
        
        inputs = Input(input_dims, name = 'main_input') 
        
        enc_lvl1_block1 = myConv3DBlock(input_tensor        = inputs, 
                                        filters             = min_filters_per_layer, 
                                        kernel_size         = lvl1_filters, 
                                        activation          = encoder_act_function,
                                        kernel_initializer  = init, 
                                        padding             = 'same', 
                                        block_name          = 'enc_lvl1a',
                                        kernel_regularizer  = None,
                                        use_batch_norm      = False,
                                        dropout_rate        = lvl1_dropout,
                                        )

        max_pool_1to2  = MaxPooling3D(pool_size = (4, 4, 4), 
                              name = 'max_pool_1to2'
                              )(enc_lvl1_block1)
                                      

    # -----------------------------------
    #         ENCODER - LEVEL 2
    # -----------------------------------

        enc_lvl2_block1 = myConv3DBlock(input_tensor        = max_pool_1to2,
                                        filters             = 2*min_filters_per_layer, 
                                        kernel_size         = lvl2_filters, 
                                        activation          = encoder_act_function,
                                        kernel_initializer  = init, 
                                        padding             = 'same', 
                                        block_name          = 'enc_lvl2a',
                                        kernel_regularizer  = lvl2_reg,
                                        use_batch_norm      = False,
                                        dropout_rate        = lvl2_dropout,
                                        )  
    

    with tf.device('/gpu:1'):
    
        enc_lvl2_block2 = myConv3DBlock(input_tensor        = enc_lvl2_block1,
                                        filters             = 2*min_filters_per_layer, 
                                        kernel_size         = lvl2_filters, 
                                        activation          = encoder_act_function,
                                        kernel_initializer  = init, 
                                        padding             = 'same', 
                                        block_name          = 'enc_lvl2b',
                                        kernel_regularizer  = lvl2_reg,
                                        use_batch_norm      = False,
                                        dropout_rate        = lvl2_dropout,
                                        )   
    
        max_pool_2to3  = MaxPooling3D(pool_size = (2, 2, 2), 
                                      name = 'max_pool_2to3'
                                      )(enc_lvl2_block2)  
          

    # -----------------------------------
    #         BOTTLENECK LAYER
    # -----------------------------------
                                    
        bottleneck_block1 = myConv3DBlock(input_tensor        = max_pool_2to3,
                                          filters             = 4*min_filters_per_layer, 
                                          kernel_size         = lvl3_filters, 
                                          activation          = encoder_act_function,
                                          kernel_initializer  = init, 
                                          padding             = 'same', 
                                          block_name          = 'bottleneck_a',
                                          kernel_regularizer  = lvl3_reg,
                                          use_batch_norm      = False,
                                          dropout_rate        = lvl3_dropout,
                                          )
                                        
        bottleneck_block2 = myConv3DBlock(input_tensor        = bottleneck_block1,
                                          filters             = 4*min_filters_per_layer, 
                                          kernel_size         = lvl3_filters, 
                                          activation          = encoder_act_function,
                                          kernel_initializer  = init, 
                                          padding             = 'same', 
                                          block_name          = 'bottleneck_b',
                                          kernel_regularizer  = lvl3_reg,
                                          use_batch_norm      = False,
                                          dropout_rate        = lvl3_dropout,
                                          ) 
    
        bottleneck_block3 = myConv3DBlock(input_tensor        = bottleneck_block2,
                                          filters             = 4*min_filters_per_layer, 
                                          kernel_size         = lvl3_filters, 
                                          activation          = encoder_act_function,
                                          kernel_initializer  = init, 
                                          padding             = 'same', 
                                          block_name          = 'bottleneck_c',
                                          kernel_regularizer  = lvl3_reg,
                                          use_batch_norm      = False,
                                          dropout_rate        = lvl3_dropout,
                                          ) 
    

    # -----------------------------------
    #         DECODER - LEVEL 2
    # -----------------------------------

        up_conv_3to2 = myUpConv3DBlock(input_tensor         = bottleneck_block3,
                                       filters              = 2*min_filters_per_layer,
                                       kernel_size          = (2, 2, 2),
                                       strides              = (2, 2, 2),
                                       kernel_initializer   = init,
                                       padding              = 'same',
                                       block_name           = 'up_conv_3to2',
                                       kernel_regularizer   = lvl2_reg,
                                       use_batch_norm       = False,
                                       dropout_rate         = lvl2_dropout,
                                       )

    with tf.device('/gpu:2'):

        skip_conn_lvl2 = Add(name = 'lvl2_longskip')([enc_lvl2_block2, up_conv_3to2])

        dec_lvl2_block1 = myConv3DBlock(input_tensor        = skip_conn_lvl2,
                                        filters             = 2*min_filters_per_layer, 
                                        kernel_size         = lvl2_filters, 
                                        activation          = encoder_act_function,
                                        kernel_initializer  = init, 
                                        padding             = 'same', 
                                        block_name          = 'dec_lvl2a',
                                        kernel_regularizer  = lvl2_reg,
                                        use_batch_norm      = False,
                                        dropout_rate        = lvl2_dropout,
                                        )
                                        
        dec_lvl2_block2 = myConv3DBlock(input_tensor        = dec_lvl2_block1,
                                        filters             = 2*min_filters_per_layer, 
                                        kernel_size         = lvl2_filters, 
                                        activation          = encoder_act_function,
                                        kernel_initializer  = init, 
                                        padding             = 'same', 
                                        block_name          = 'dec_lvl2b',
                                        kernel_regularizer  = lvl2_reg,
                                        use_batch_norm      = False,
                                        dropout_rate        = lvl2_dropout,
                                        )



    # -----------------------------------
    #         DECODER - LEVEL 1
    # -----------------------------------

        up_conv_2to1 = myUpConv3DBlock(input_tensor         = dec_lvl2_block2,
                                       filters              = min_filters_per_layer,
                                       kernel_size          = (4, 4, 4),
                                       strides              = (4, 4, 4),
                                       kernel_initializer   = init,
                                       padding              = 'same',
                                       block_name           = 'up_conv_2to1',
                                       kernel_regularizer   = None,
                                       use_batch_norm       = False,
                                       dropout_rate         = lvl2_dropout,
                                       )

        skip_conn_lvl1 = Add(name = 'lvl1_longskip')([enc_lvl1_block1, up_conv_2to1])

    with tf.device('/gpu:3'):
    
        dec_lvl1_block1 = myConv3DBlock(input_tensor        = skip_conn_lvl1,
                                        filters             = min_filters_per_layer, 
                                        kernel_size         = lvl1_filters, 
                                        activation          = encoder_act_function,
                                        kernel_initializer  = init, 
                                        padding             = 'same', 
                                        block_name          = 'dec_lvl1a',
                                        kernel_regularizer  = None,
                                        use_batch_norm      = False,
                                        dropout_rate        = lvl1_dropout,
                                        )
                                        
    
        outputs = Conv3D(filters            = num_classes,
                         kernel_size        = (1, 1, 1),
                         activation         = classification_act_function,
                         kernel_initializer = init,
                         name               = 'main_output',
                         )(dec_lvl1_block1)


    # define the model object and the optimizer
    model   = Model(inputs = [inputs], outputs = [outputs])
    my_adam = keras.optimizers.Adam(lr = learning_rate, beta_1 = 0.9, beta_2 = 0.999)
    
    # compile the model
    model.compile(loss      = losses_dict[loss_function],
                  optimizer = my_adam,
                  metrics   = ['categorical_crossentropy',
                               losses_dict['multiclass_dice_coeff'],
                               losses_dict['average_multiclass_dice_coeff'],
                               losses_dict['tanimoto_coefficient'] ])
    
    
    # return the model object
    return model


## -----------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------

# Three layers CNN, max-pooling replaced by strided conv.s, first str. conv.s pooling has factor 4

def ThreeLevelsStrConv(input_dims,
                       num_classes,
                       init = 'glorot_normal',
                       encoder_act_function = 'elu',
                       decoder_act_function = 'relu',
                       classification_act_function = 'softmax', 
                       loss_function = 'categorical_crossentropy',
                       learning_rate = 1e-05,
                       min_filters_per_layer = 16,
                       use_kernel_reg = False,
                       use_dropout = False,
                       ):

    if use_kernel_reg == True:
        lvl2_reg = l2(0.005)
        lvl3_reg = l2(0.01)
    else:
        lvl2_reg = None
        lvl3_reg = None
        
    if use_dropout == True:
        lvl1_dropout = 0.0 
        lvl2_dropout = 0.25
        lvl3_dropout = 0.5
    else:
        lvl1_dropout = 0.0 
        lvl2_dropout = 0.0
        lvl3_dropout = 0.0

    lvl1_filters = (3, 3, 3)
    lvl2_filters = (3, 3, 3)
    lvl3_filters = (3, 3, 3)

    # -----------------------------------
    #         ENCODER - LEVEL 1
    # -----------------------------------

    with tf.device('/gpu:0'):
        
        inputs = Input(input_dims, name = 'main_input') 
        
        enc_lvl1_block1 = myConv3DBlock(input_tensor        = inputs, 
                                        filters             = min_filters_per_layer, 
                                        kernel_size         = lvl1_filters, 
                                        activation          = encoder_act_function,
                                        kernel_initializer  = init, 
                                        padding             = 'same', 
                                        block_name          = 'enc_lvl1a',
                                        kernel_regularizer  = None,
                                        use_batch_norm      = False,
                                        dropout_rate        = lvl1_dropout,
                                        )

        down_conv_1to2  = myDownConv3DBlock(input_tensor        = enc_lvl1_block1, 
                                            filters             = 2*min_filters_per_layer, 
                                            kernel_size         = (4, 4, 4), 
                                            strides             = (4, 4, 4),
                                            activation          = encoder_act_function,
                                            kernel_initializer  = init, 
                                            padding             = 'same', 
                                            block_name          = 'down_conv_1to2',
                                            kernel_regularizer  = None,
                                            use_batch_norm      = False,
                                            dropout_rate        = lvl1_dropout,
                                            )

    # -----------------------------------
    #         ENCODER - LEVEL 2
    # -----------------------------------

                             
        enc_lvl2_block1 = myConv3DBlock(input_tensor        = down_conv_1to2,
                                        filters             = 2*min_filters_per_layer, 
                                        kernel_size         = lvl2_filters, 
                                        activation          = encoder_act_function,
                                        kernel_initializer  = init, 
                                        padding             = 'same', 
                                        block_name          = 'enc_lvl2a',
                                        kernel_regularizer  = lvl2_reg,
                                        use_batch_norm      = False,
                                        dropout_rate        = lvl2_dropout,
                                        )  
        
    with tf.device('/gpu:1'):
    
        enc_lvl2_block2 = myConv3DBlock(input_tensor        = enc_lvl2_block1,
                                        filters             = 2*min_filters_per_layer, 
                                        kernel_size         = lvl2_filters, 
                                        activation          = encoder_act_function,
                                        kernel_initializer  = init, 
                                        padding             = 'same', 
                                        block_name          = 'enc_lvl2b',
                                        kernel_regularizer  = lvl2_reg,
                                        use_batch_norm      = False,
                                        dropout_rate        = lvl2_dropout,
                                        )   
        
        down_conv_2to3  = myDownConv3DBlock(input_tensor        = enc_lvl2_block2,
                                            filters             = 4*min_filters_per_layer, 
                                            kernel_size         = (2, 2, 2), 
                                            strides             = (2, 2, 2),
                                            activation          = encoder_act_function,
                                            kernel_initializer  = init, 
                                            padding             = 'same', 
                                            block_name          = 'down_conv_2to3',
                                            kernel_regularizer  = None,
                                            use_batch_norm      = False,
                                            dropout_rate        = lvl2_dropout,
                                            )
                                      


    # -----------------------------------
    #         BOTTLENECK LAYER
    # -----------------------------------
                                    
        bottleneck_block1 = myConv3DBlock(input_tensor        = down_conv_2to3,
                                          filters             = 4*min_filters_per_layer, 
                                          kernel_size         = lvl3_filters, 
                                          activation          = encoder_act_function,
                                          kernel_initializer  = init, 
                                          padding             = 'same', 
                                          block_name          = 'bottleneck_a',
                                          kernel_regularizer  = lvl3_reg,
                                          use_batch_norm      = False,
                                          dropout_rate        = lvl3_dropout
                                          )
        
        bottleneck_block2 = myConv3DBlock(input_tensor        = bottleneck_block1,
                                          filters             = 4*min_filters_per_layer, 
                                          kernel_size         = lvl3_filters, 
                                          activation          = encoder_act_function,
                                          kernel_initializer  = init, 
                                          padding             = 'same', 
                                          block_name          = 'bottleneck_b',
                                          kernel_regularizer  = lvl3_reg,
                                          use_batch_norm      = False,
                                          dropout_rate        = lvl3_dropout
                                          ) 

        bottleneck_block3 = myConv3DBlock(input_tensor        = bottleneck_block2,
                                          filters             = 4*min_filters_per_layer, 
                                          kernel_size         = lvl3_filters, 
                                          activation          = encoder_act_function,
                                          kernel_initializer  = init, 
                                          padding             = 'same', 
                                          block_name          = 'bottleneck_c',
                                          kernel_regularizer  = lvl3_reg,
                                          use_batch_norm      = False,
                                          dropout_rate        = lvl3_dropout
                                          )
    
    # -----------------------------------
    #         DECODER - LEVEL 2
    # -----------------------------------

        
        up_conv_3to2 = myUpConv3DBlock(input_tensor         = bottleneck_block3,
                                       filters              = 2*min_filters_per_layer,
                                       kernel_size          = (2, 2, 2),
                                       strides              = (2, 2, 2),
                                       kernel_initializer   = init,
                                       padding              = 'same',
                                       block_name           = 'up_conv_3to2',
                                       kernel_regularizer   = lvl2_reg,
                                       use_batch_norm       = False,
                                       dropout_rate         = lvl2_dropout)

    with tf.device('/gpu:2'):

        skip_conn_lvl2 = Add(name = 'lvl2_longskip')([enc_lvl2_block2, up_conv_3to2])

        dec_lvl2_block1 = myConv3DBlock(input_tensor        = skip_conn_lvl2,
                                        filters             = 2*min_filters_per_layer, 
                                        kernel_size         = lvl2_filters, 
                                        activation          = encoder_act_function,
                                        kernel_initializer  = init, 
                                        padding             = 'same', 
                                        block_name          = 'dec_lvl2a',
                                        kernel_regularizer  = lvl2_reg,
                                        use_batch_norm      = False,
                                        dropout_rate        = lvl2_dropout)

        dec_lvl2_block2 = myConv3DBlock(input_tensor        = dec_lvl2_block1,
                                        filters             = 2*min_filters_per_layer, 
                                        kernel_size         = lvl2_filters, 
                                        activation          = encoder_act_function,
                                        kernel_initializer  = init, 
                                        padding             = 'same', 
                                        block_name          = 'dec_lvl2b',
                                        kernel_regularizer  = lvl2_reg,
                                        use_batch_norm      = False,
                                        dropout_rate        = lvl2_dropout)

    # -----------------------------------
    #         DECODER - LEVEL 1
    # -----------------------------------

    
        up_conv_2to1 = myUpConv3DBlock(input_tensor         = dec_lvl2_block2,
                                       filters              = min_filters_per_layer,
                                       kernel_size          = (4, 4, 4),
                                       strides              = (4, 4, 4),
                                       kernel_initializer   = init,
                                       padding              = 'same',
                                       block_name           = 'up_conv_2to1',
                                       kernel_regularizer   = None,
                                       use_batch_norm       = False,
                                       dropout_rate         = lvl2_dropout)

        skip_conn_lvl1 = Add(name = 'lvl1_longskip')([enc_lvl1_block1, up_conv_2to1])

    with tf.device('/gpu:3'):
    
        dec_lvl1_block1 = myConv3DBlock(input_tensor        = skip_conn_lvl1,
                                        filters             = min_filters_per_layer, 
                                        kernel_size         = lvl1_filters, 
                                        activation          = encoder_act_function,
                                        kernel_initializer  = init, 
                                        padding             = 'same', 
                                        block_name          = 'dec_lvl1a',
                                        kernel_regularizer  = None,
                                        use_batch_norm      = False,
                                        dropout_rate        = lvl1_dropout)    

        outputs = Conv3D(filters            = num_classes,
                         kernel_size        = (1, 1, 1),
                         activation         = classification_act_function,
                         kernel_initializer = init,
                         name               = 'main_output',
                         )(dec_lvl1_block1)


    # define the model object and the optimizer
    model   = Model(inputs=[inputs], outputs=[outputs])
    my_adam = keras.optimizers.Adam(lr = learning_rate, beta_1 = 0.9, beta_2 = 0.999)
    
        
    # compile the actual model
    #model.compile(loss = loss_function, optimizer = my_adam, metrics=['categorical_crossentropy'])
    
    model.compile(loss      = losses_dict[loss_function],
                  optimizer = my_adam,
                  metrics   = ['categorical_crossentropy',
                               losses_dict['multiclass_dice_coeff'],
                               losses_dict['average_multiclass_dice_coeff'],
                               #losses_dict['weighted_multiclass_dice_coeff'],
                               losses_dict['tanimoto_coefficient'] ])
    
    # print a summary of the model
    model.summary()
    
    # return the model object
    return model

