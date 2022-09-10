import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import numpy as np
from datasets import dataset_mean, dataset_std
from datasets.cityscapes import CityscapesGenerator
from utils import print_np
from matplotlib import pyplot as plt
import time

val_freq = 200


def unnormalize(img, mean, std):
    unnorm_img = np.zeros(img.shape)
    unnorm_img[:, :, :, 0] = img[:, :, :, 0]*std[0] + mean[0]
    unnorm_img[:, :, :, 1] = img[:, :, :, 1]*std[1] + mean[1]
    unnorm_img[:, :, :, 2] = img[:, :, :, 2]*std[2] + mean[2]

    return (unnorm_img*255).astype(np.uint8)


class TensorboardCallback(tf.keras.callbacks.Callback):
    def __init__(self, logdir, val_data, monitor, hparams):
        super(TensorboardCallback).__init__() 
        self.logdir = logdir
        self.val_data = val_data
        self.monitor = monitor
        self.hparams = hparams
        self.best = 0
    
    def on_train_begin(self, logs=None):
        # Initialize the best as 0.
        self.best = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if np.greater(current, self.best):
            img, _ = self.val_data[0]
            label = self.val_data.y_20
            img_pred = self.model.predict(img)

            # Unnormalize the images
            img = unnormalize(img, dataset_mean, dataset_std)

            # Write data to Tensorboard
            label = CityscapesGenerator.from_one_hot_to_rgb_bkup(label)
            prediction = CityscapesGenerator.from_one_hot_to_rgb_bkup(img_pred)

            img_concat = np.concatenate((img, np.concatenate((label, prediction), axis=2)), axis=2)

            self.hparams["Learning rate"] = self.model.optimizer.lr

            # Creates a file writer for the log directory.
            file_writer = tf.summary.create_file_writer(self.logdir)

            # Using the file writer, log the reshaped image.
            with file_writer.as_default():
                tf.summary.image("Training Data", img_concat, step=epoch)
                # hp.hparams_config(
                #                     hparams=self.hparams,
                #                     # metrics=[hp.Metric(self.monitor, display_name='MIoU')],
                #                     metrics=[hp.Metric('accuracy', display_name='Accuracy')]
                #                 )


class VisualizeImagesCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset_data):
        super(VisualizeImagesCallback).__init__() 
        self.dataset_data = dataset_data
   
    def on_batch_end(self, epoch, logs=None):
        img = self.dataset_data[0][0]
        label = self.dataset_data.y_20
        label = CityscapesGenerator.from_one_hot_to_rgb_bkup(label)
        img_pred = self.model.predict(img)
        prediction = CityscapesGenerator.from_one_hot_to_rgb_bkup(img_pred)

        # Unnormalize the images
        img = unnormalize(img, dataset_mean, dataset_std)
        
        plt.figure(2, figsize=(20,4))
        plt.subplot(1, 3, 1)
        plt.imshow(img[0])
        plt.subplot(1, 3, 2)
        plt.imshow(label[0])
        plt.subplot(1, 3, 3)
        plt.imshow(prediction[0])
        plt.pause(1e-9)
