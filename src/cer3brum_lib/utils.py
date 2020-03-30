## ----------------------------------------
## Mixed utilities
## ----------------------------------------
## 
## ----------------------------------------
## Author: Dennis Bontempi, Michele Svanera
## Version: 2.0
## Email: dennis.bontempi@glasgow.ac.uk
## Status: ready to use
## Modified: 20 Feb 19
## ----------------------------------------


import os
import shutil
import tensorflow as tf

from keras import backend as K
from keras.callbacks import TensorBoard


## -----------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------

class TrainValTensorBoard(TensorBoard):
    
    
    def __init__(self, model_name, log_dir, **kwargs):
        # make the original `TensorBoard` log to a subdirectory 'training'
        
        model_dir = os.path.join(log_dir, model_name[0:-3])
        
        # if a tb log for the model exists already, remove it
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir, ignore_errors=True)
        
        training_log_dir = os.path.join(log_dir, model_name[0:-3], 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)
    
        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, model_name[0:-3], 'validation')
    
    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)
    
    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()
    
    # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)
    
    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

## -----------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------

"""
    Exploiting the list_local_devices function belonging to the TF device_lib, get the number of 
    GPUs currently available on the machine (useful to define the upper bound of the gpu argument
    in both training and prediction phase scripts).
    
"""

from tensorflow.python.client import device_lib

def NumAvailableGPUs():
    availableDevices = device_lib.list_local_devices()
    
    
    # list_local_devices() returns the name of a device, the device type, the memory limit and a 
    # couple of other information. Both name and device_type contain the information we're searching
    # for, but device_type can either be "CPU" or "GPU", so go for that.
    
    return len([device.name for device in availableDevices if device.device_type == 'GPU'])
    
    
    
    
    
    
    
    
    

