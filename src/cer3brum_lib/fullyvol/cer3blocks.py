
## ----------------------------------------
## Useful Keras blocks definition (fullyvol)
## ----------------------------------------
## 
## ----------------------------------------
## Author: Dennis Bontempi, Michele Svanera
## Version: 2.0
## Email: dennis.bontempi@glasgow.ac.uk
## Status: ready to use
## Modified: 20 Feb 19
## ----------------------------------------


from keras.regularizers import l1, l2

from keras.layers import Dense, Activation, BatchNormalization
from keras.layers.core import Dropout, Lambda, Activation
from keras.layers.pooling import MaxPooling2D, MaxPooling3D
from keras.layers.convolutional import Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose


## -----------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------


def myConv3DBlock(input_tensor,
                  filters,
                  kernel_size,
                  activation,
                  kernel_initializer,
                  padding,
                  block_name,
                  kernel_regularizer,
                  use_batch_norm,
                  dropout_rate):

    """
        3D convolution block exploiting Keras.
        
        This function returns a "black box" containing potentially an conv. layer, a BN layer 
        and a dropout layer. The scheme is:
        
         ... -> Conv 3D -> Activation -> BatchNorm -> Dropout -> ...
            
    """

    output_tensor = Conv3D(filters              = filters, 
                           kernel_size          = kernel_size, 
                           activation           = activation,
                           kernel_initializer   = kernel_initializer, 
                           padding              = padding, 
                           name                 = block_name + '_conv',
                           kernel_regularizer   = kernel_regularizer
                           )(input_tensor) 
    
    if use_batch_norm:        
        output_tensor = BatchNormalization(name = block_name + '_bn')(output_tensor)

    if dropout_rate > 0.0:
        output_tensor = Dropout(rate            = dropout_rate, 
                                name            = block_name + '_dr'
                                )(output_tensor)
    
    return output_tensor
    
## -----------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------

def myDownConv3DBlock(input_tensor,
                      filters,
                      kernel_size,
                      activation,
                      strides,
                      kernel_initializer,
                      padding,
                      block_name,
                      kernel_regularizer,
                      use_batch_norm,
                      dropout_rate):

    """
        3D strided convolution block exploiting Keras.
        
        This function returns a "black box" containing potentially an strided (down) conv. layer,
        a BN layer and a dropout layer. The scheme is:
        
         ... -> Conv 3D -> Activation -> BatchNorm -> Dropout -> ...
            
    """

    output_tensor = Conv3D(filters              = filters, 
                           kernel_size          = kernel_size, 
                           strides              = strides,
                           activation           = activation,
                           kernel_initializer   = kernel_initializer, 
                           padding              = padding, 
                           name                 = block_name,
                           kernel_regularizer   = kernel_regularizer
                           )(input_tensor) 
    
    if use_batch_norm:        
        output_tensor = BatchNormalization(name = block_name + '_bn')(output_tensor)

    if dropout_rate > 0.0:
        output_tensor = Dropout(rate            = dropout_rate, 
                                name            = block_name + '_dr'
                                )(output_tensor)
    
    return output_tensor
    
    
## -----------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------
 
    
def myUpConv3DBlock(input_tensor,
                    filters,
                    kernel_size,
                    strides,
                    kernel_initializer,
                    padding,
                    block_name,
                    kernel_regularizer,
                    use_batch_norm,
                    dropout_rate):

    """
        3D transpose convolution block exploiting Keras.
        
        This function returns a "black box" containing potentially an up conv. layer, a BN layer 
        and a dropout layer. The scheme is:
        
         ... -> Transpose Conv 3D -> Activation -> BatchNorm -> Dropout -> ...
            
    """
    
    # NOTES:
    # 1. activation is by default "None" in Conv3DTranspose;
    
    output_tensor = Conv3DTranspose(filters              = filters, 
                                    kernel_size          = kernel_size, 
                                    strides              = strides,
                                    kernel_initializer   = kernel_initializer, 
                                    padding              = padding, 
                                    name                 = block_name,
                                    kernel_regularizer   = kernel_regularizer
                                    )(input_tensor) 
    
    if use_batch_norm:        
        output_tensor = BatchNormalization(name = block_name + '_bn')(output_tensor)

    if dropout_rate > 0.0:
        output_tensor = Dropout(rate            = dropout_rate, 
                                name            = block_name + '_dr'
                                )(output_tensor)
    
    return output_tensor
    
    
## -----------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------

