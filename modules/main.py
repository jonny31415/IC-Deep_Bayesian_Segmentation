#!/usr/bin/env python3

import os
from datetime import datetime
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import numpy as np
from models.custom_models import BasicSegmentationHead, BayesianSegmentationHeadAleatoric, BayesianSegmentationHeadEpistemic, BayesianSegmentationHeadFullProbOneConv, BayesianSegmentationHeadFullProb
from metrics.onehot_miou import OneHotMeanIoU, MyMeanIOU
from datasets.cityscapes import CityscapesGenerator
from utils import print_np, get_layers, get_summary, plot_predictions, plot_uncertainty_surface
from callbacks.callbacks import TensorboardCallback, VisualizeImagesCallback
from losses.NLL import NLL
from trunks.ResNet import get_resnet
from trunks.DeepLabV3Plus import getDeeplabV3Plus
from trunks.HRNetOCR import getHRNetOCR
import matplotlib.pyplot as plt
from datasets import dataset_mean, dataset_std
from time import time

# Disable memory growth
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.experimental.set_memory_growth(physical_devices[1], True)

# os.environ["CUDA_VISIBLE_DEVICES"]="0" # first gpu
os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # runs in cpu

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' 

############
## PARAMS ##
############

ONE_IMG_PATH = os.path.join('/media', 'olorin', 'Documentos', 'datasets', 'cityscapes', 'leftImg8bit', 'train', 'aachen')
ONE_LABEL_PATH = os.path.join('/media', 'olorin', 'Documentos', 'datasets', 'cityscapes', 'gtFine', 'train', 'aachen')

TRAIN_INPUT_DIR = os.path.join('/media', 'olorin', 'Documentos', 'datasets', 'cityscapes', 'leftImg8bit', 'train')
TRAIN_TARGET_DIR = os.path.join('/media', 'olorin', 'Documentos', 'datasets', 'cityscapes', 'gtFine', 'train')
VAL_INPUT_DIR = os.path.join('/media', 'olorin', 'Documentos', 'datasets', 'cityscapes', 'leftImg8bit', 'val')
VAL_TARGET_DIR = os.path.join('/media', 'olorin', 'Documentos', 'datasets', 'cityscapes', 'gtFine', 'val')
TEST_INPUT_DIR = os.path.join('/media', 'olorin', 'Documentos', 'datasets', 'cityscapes', 'leftImg8bit', 'test')
TEST_TARGET_DIR = os.path.join('/media', 'olorin', 'Documentos', 'datasets', 'cityscapes', 'gtFine', 'test')
BATCH_SIZE = 2
N_CLASSES = 19
ADD_IGNORE_CLASS = True

if ADD_IGNORE_CLASS:
    N_CLASSES += 1

INITIAL_LR = 1e-3
HEIGHT = 512
WIDTH = 1024
CHANNELS = 3
IMG_SIZE = (HEIGHT, WIDTH)
PRETRAINED = True
DATA_AUG = False
MAX_EPOCHS = 200
SINGLE_IMAGE_MODE = False
VISUALIZE_IMAGES = False
TENSORBOARD_WRITE = False
POLY_EXP = 2
TRUNK_NAME = 'ResNet50'
# TRUNK_NAME = 'DeeplabV3Plus'
# TRUNK_NAME = 'HRNetOCR'
MODEL_NAME = 'BayesianSegHeadAleatoric'
sch = 'exp'
# sch = 'poly'
VALIDATION = True
TRAIN_SAVE_FREQ_EPOCHS = 100
VAL_PATIENCE = 20


def get_trunk(input_shape, selected_trunk):
    if selected_trunk == 'ResNet50':
        trunk = get_resnet(input_shape, selected_trunk, output_stride=None)
    elif selected_trunk == 'DeeplabV3Plus':
        trunk = getDeeplabV3Plus(input_shape, N_CLASSES)
    elif selected_trunk == 'HRNetOCR':
        trunk = getHRNetOCR(input_shape, N_CLASSES)
    else:  
        raise NotImplementedError
 
    return trunk, trunk.output


def get_model(input_shape, train_gen_len, selected_model, selected_trunk):
    trunk, trunk_features = get_trunk(input_shape, selected_trunk)

    if selected_model == 'BasicSegHead':
        segmentation_head = BasicSegmentationHead(N_CLASSES, input_shape=trunk_features.shape, output_shape=(None,) + input_shape)
        segmentation_output = trunk_features

        for layer in segmentation_head.layers:
            segmentation_output = layer(segmentation_output)

        segmentation_output = tf.keras.layers.Softmax()(segmentation_output)
    
        model = tf.keras.Model(inputs=trunk.input, outputs=segmentation_output)

        loss_fn = tf.keras.losses.CategoricalCrossentropy(name="CategoricalCrossentropy")

    elif selected_model == 'BayesianSegHeadEpistemic':
        segmentation_head = BayesianSegmentationHeadEpistemic(N_CLASSES, train_gen_len=train_gen_len, use_exact_kl=False)
        segmentation_output = trunk_features

        for layer in segmentation_head.layers:
            segmentation_output = layer(segmentation_output)

        segmentation_output = tf.keras.layers.Softmax()(segmentation_output)
    
        model = tf.keras.Model(inputs=trunk.input, outputs=segmentation_output)

        loss_fn = tf.keras.losses.CategoricalCrossentropy(name="CategoricalCrossentropy")
    
    elif selected_model == 'BayesianSegHeadAleatoric':
        segmentation_head = BayesianSegmentationHeadAleatoric(N_CLASSES, train_gen_len=train_gen_len, use_exact_kl=False)
        segmentation_output = trunk_features

        for layer in segmentation_head.layers:
            segmentation_output = layer(segmentation_output)
    
        model = tf.keras.Model(inputs=trunk.input, outputs=segmentation_output)

        loss_fn = NLL
    
    elif selected_model == 'BayesianSegHeadFullProbOneConv':
        segmentation_head = BayesianSegmentationHeadFullProbOneConv(N_CLASSES, train_gen_len=train_gen_len, use_exact_kl=False)
        segmentation_output = trunk_features

        for layer in segmentation_head.layers:
            segmentation_output = layer(segmentation_output)
    
        model = tf.keras.Model(inputs=trunk.input, outputs=segmentation_output)

        loss_fn = NLL
    
    elif selected_model == 'BayesianSegHeadFullProb':
        segmentation_head = BayesianSegmentationHeadFullProb(N_CLASSES, train_gen_len=train_gen_len, use_exact_kl=False)
        segmentation_output = trunk_features

        for layer in segmentation_head.layers:
            segmentation_output = layer(segmentation_output)
    
        model = tf.keras.Model(inputs=trunk.input, outputs=segmentation_output)

        loss_fn = NLL

    else:
        raise NotImplementedError

    return model, loss_fn


def poly_lr_schedule(epoch, initial_learning_rate=1e-2):
        return initial_learning_rate*tf.pow(1 - epoch / MAX_EPOCHS, POLY_EXP)


def get_scheduler(sch, decay_step):
    lr_schedule = None
    if sch == 'exp':
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=INITIAL_LR,
        decay_steps=decay_step,
        decay_rate=0.9)
    elif sch == 'poly':
        lr_schedule = poly_lr_schedule
    else:
        raise ValueError


    return lr_schedule


def unnormalize(img, mean, std):
    unnorm_img = np.zeros(img.shape)
    unnorm_img[:, :, :, 0] = img[:, :, :, 0]*std[0] + mean[0]
    unnorm_img[:, :, :, 1] = img[:, :, :, 1]*std[1] + mean[1]
    unnorm_img[:, :, :, 2] = img[:, :, :, 2]*std[2] + mean[2]

    return (unnorm_img*255).astype(np.uint8)


class MyMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)


def main():
    input_shape = (HEIGHT, WIDTH, CHANNELS)

    # --- Train
    train_input_img_paths = sorted(
        [
            os.path.join(TRAIN_INPUT_DIR, city, fname)
            for city in os.listdir(TRAIN_INPUT_DIR)
                for fname in os.listdir(os.path.join(TRAIN_INPUT_DIR, city))
                    if fname.endswith(".png")
        ]
    )
    train_target_img_paths = sorted(
        [
            os.path.join(TRAIN_TARGET_DIR, city, fname)
            for city in os.listdir(TRAIN_TARGET_DIR)
                for fname in os.listdir(os.path.join(TRAIN_TARGET_DIR, city))
                    if fname.endswith("gtFine_labelIds.png") and not fname.startswith(".")
        ]
    )
    # ---

    # --- Validation
    val_input_img_paths = sorted(
        [
            os.path.join(VAL_INPUT_DIR, city, fname)
            for city in os.listdir(VAL_INPUT_DIR)
                for fname in os.listdir(os.path.join(VAL_INPUT_DIR, city))
                    if fname.endswith(".png")
        ]
    )
    val_target_img_paths = sorted(
        [
            os.path.join(VAL_TARGET_DIR, city, fname)
            for city in os.listdir(VAL_TARGET_DIR)
                for fname in os.listdir(os.path.join(VAL_TARGET_DIR, city))
                    if fname.endswith("gtFine_labelIds.png") and not fname.startswith(".")
        ]
    )
    # ---

    one_img_path = sorted(
        [
            os.path.join(ONE_IMG_PATH, fname)
            for fname in os.listdir(os.path.join(ONE_IMG_PATH))
                if fname.endswith(".png")
        ]
    )

    one_label_path = sorted(
        [
            os.path.join(ONE_LABEL_PATH, fname)
            for fname in os.listdir(os.path.join(ONE_LABEL_PATH))
                if fname.endswith("gtFine_labelIds.png") and not fname.startswith(".")
        ]
    )

    if not SINGLE_IMAGE_MODE:
        # Creates train dataset
        train_gen = CityscapesGenerator(BATCH_SIZE, IMG_SIZE, train_input_img_paths, train_target_img_paths, 
                                        data_aug=DATA_AUG, add_ignore_class=ADD_IGNORE_CLASS)

        # Creates validation dataset
        val_gen = CityscapesGenerator(BATCH_SIZE, IMG_SIZE, val_input_img_paths, val_target_img_paths, add_ignore_class=ADD_IGNORE_CLASS)
    
        train_dataset = train_gen
        validation_dataset = val_gen
        decay_step = 5
    else:
        one_img_test = one_img_test[0:1]*200
        one_label_test = one_label_test[0:1]*200
        train_one_img_gen = CityscapesGenerator(2, IMG_SIZE, one_img_path, one_label_path, 
                                                data_aug=DATA_AUG, add_ignore_class=ADD_IGNORE_CLASS)
        val_one_img_gen = CityscapesGenerator(2, IMG_SIZE, one_img_path, one_label_path, add_ignore_class=ADD_IGNORE_CLASS)

        train_dataset = train_one_img_gen
        validation_dataset = val_one_img_gen
        decay_step = 50

    model, loss_fn = get_model(input_shape, train_gen_len=len(train_dataset), selected_model=MODEL_NAME, selected_trunk=TRUNK_NAME)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=INITIAL_LR, momentum=0.9), loss=loss_fn, 
                    metrics=['accuracy', OneHotMeanIoU(N_CLASSES)])

    if PRETRAINED:
        # model.load_weights("weights/ResNet50-BayesianSegHeadAleatoric-2022-08-04 09:25:57.287341/056-0.70.h5", by_name=True, skip_mismatch=True)
        model.load_weights("weights/ResNet50-BayesianSegHeadAleatoric-2022-08-12 11:13:23.842612/133-0.55.h5", by_name=True, skip_mismatch=True)

    # Learning rate decay schedule
    lr_schedule = get_scheduler(sch, decay_step)

    datenow = str(datetime.now())
    params_dict = {"Optimizer": 'SGD', "Loss": loss_fn.__name__, "Arch": MODEL_NAME, "Trunk": TRUNK_NAME,
                   "Batch size": BATCH_SIZE}
    # params_dict = {"Optimizer": 'SGD', "Loss": model.loss.name, "Arch": MODEL_NAME, "Trunk": TRUNK_NAME,
    #                "Batch size": BATCH_SIZE}
    
    my_callbacks = [
        tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=True), 
    ]
    
    if VISUALIZE_IMAGES:
        visualize_images_cb = VisualizeImagesCallback(train_dataset)
        my_callbacks.append(visualize_images_cb)

    if TENSORBOARD_WRITE:
        tensorboard_write_cb = [tf.keras.callbacks.TensorBoard(log_dir=f'./logs/{TRUNK_NAME}-{MODEL_NAME}'+'-'+ datenow, 
                                write_images=True, update_freq=10),
        TensorboardCallback(logdir=f'./logs/{TRUNK_NAME}-{MODEL_NAME}'+'-'+ datenow +'/Images/', val_data=train_dataset,
                            monitor="one_hot_mean_io_u", hparams=params_dict), 
                            hp.KerasCallback(writer=f'./logs/{TRUNK_NAME}-{MODEL_NAME}'+'-'+ datenow, hparams=params_dict)]
        my_callbacks.extend(tensorboard_write_cb)

    # if VALIDATION:
    #     # EarlyStopping for validation
    #     my_callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_one_hot_mean_io_u", patience=VAL_PATIENCE, mode="max"))
    #     model_save_cb = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('./weights/', TRUNK_NAME+'-'+MODEL_NAME+'-'+ datenow,
    #                                     '{epoch:03d}-{val_one_hot_mean_io_u:.2f}.h5'), monitor="val_one_hot_mean_io_u",
    #                                     mode="max", save_best_only=True, verbose=True),
    #     my_callbacks.append(model_save_cb)

    #     model.fit(x=train_dataset, epochs=MAX_EPOCHS, verbose=1, callbacks=my_callbacks,
    #             validation_split = 0.85, shuffle=True)
    # else:
    #     # Saves best weights after save_freq batches
    #     model_save_cb = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('./weights/', TRUNK_NAME+'-'+MODEL_NAME+'-'+ datenow,
    #                                     '{epoch:03d}-{one_hot_mean_io_u:.2f}.h5'), monitor="one_hot_mean_io_u",
    #                                     mode="max", save_best_only=True, verbose=True, save_freq='epoch'),
    #     my_callbacks.append(model_save_cb)
    
    #     model.fit(x=train_dataset, epochs=MAX_EPOCHS, verbose=1, callbacks=my_callbacks, shuffle=True)
    
    # Test

    from datasets import dataset_mean, dataset_std
    img, _ = validation_dataset[0]
    label = validation_dataset.y_20
    img_pred = model.predict(img)

    # Unnormalize the images
    img = unnormalize(img, dataset_mean, dataset_std)

    label = CityscapesGenerator.from_one_hot_to_rgb_bkup(label)
    prediction = CityscapesGenerator.from_one_hot_to_rgb_bkup(img_pred)

    # img_concat = np.concatenate((img[0], np.concatenate((label[0], prediction[0]), axis=1)), axis=1)
    # plt.imsave(f'Model_prediction_crop.png', img_concat)
    # plt.imsave(f'Model_prediction_crop_1.png', prediction[0])
    # model.evaluate(validation_dataset)

    # Uncertainty
    model_covmat = model(validation_dataset[0][0]).covariance()
    model_variance = tf.linalg.diag_part(model_covmat)[:, None]

    #----
    H, W = model_variance.shape[2], model_variance.shape[3]

    # Initialize the plot axes.
    fig, axs = plt.subplots(2, 2, figsize=(12, 5))

    uncertaity_reduced = tf.math.reduce_sum(model_variance, axis=-1)

    # Plots the predictive uncertainty.
    pcm_0 = plot_uncertainty_surface(uncertaity_reduced[0,0,:,:], ax=axs[1,1], shape=(H, W))

    # Adds color bars and titles.
    fig.colorbar(pcm_0, ax=axs[1,1])

    axs[0,0].axes.xaxis.set_visible(False)
    axs[0,0].axes.yaxis.set_visible(False)
    axs[0,1].axes.xaxis.set_visible(False)
    axs[0,1].axes.yaxis.set_visible(False)
    axs[1,0].axes.xaxis.set_visible(False)
    axs[1,0].axes.yaxis.set_visible(False)


    axs[0,0].imshow(img[0])
    axs[0,1].imshow(label[0])
    axs[1,0].imshow(prediction[0])
    axs[0,0].set_title(f"Image")
    axs[0,1].set_title(f"Label")
    axs[1,0].set_title(f"Prediction")
    axs[1,1].set_title(f"Predictive Uncertainty")

    plt.subplots_adjust(wspace=0)

    plt.savefig("Uncertainty_pred.png", bbox_inches='tight', pad_inches = 0.1)
    # -----

    # # Initialize the plot axes.
    # fig, axs = plt.subplots(2, 2, figsize=(12, 5))

    # # Uncertainty video
    # for i in range(len(train_input_img_paths)): 

    #     img, _ = train_dataset[i]
    #     label = train_dataset.y_20
    #     img_pred = model.predict(img)

    #     # Unnormalize the images
    #     img = unnormalize(img, dataset_mean, dataset_std)

    #     label = CityscapesGenerator.from_one_hot_to_rgb_bkup(label)
    #     prediction = CityscapesGenerator.from_one_hot_to_rgb_bkup(img_pred)

    #     model_covmat = model(train_dataset[i][0]).covariance()
    #     model_variance = tf.linalg.diag_part(model_covmat)[:, None]

    #     #----
    #     H, W = model_variance.shape[2], model_variance.shape[3]

    #     uncertaity_reduced = tf.math.reduce_sum(model_variance, axis=-1)

    #     # Plots the predictive uncertainty.
    #     pcm_0 = plot_uncertainty_surface(uncertaity_reduced[0,0,:,:], ax=axs[1,1], shape=(H, W))

    #     # Adds color bars and titles.
    #     cb = fig.colorbar(pcm_0, ax=axs[1,1])

    #     axs[0,0].axes.xaxis.set_visible(False)
    #     axs[0,0].axes.yaxis.set_visible(False)
    #     axs[0,1].axes.xaxis.set_visible(False)
    #     axs[0,1].axes.yaxis.set_visible(False)
    #     axs[1,0].axes.xaxis.set_visible(False)
    #     axs[1,0].axes.yaxis.set_visible(False)

    #     axs[0,0].imshow(img[0])
    #     axs[0,1].imshow(label[0])
    #     axs[1,0].imshow(prediction[0])
    #     axs[0,0].set_title(f"Image")
    #     axs[0,1].set_title(f"Label")
    #     axs[1,0].set_title(f"Prediction")
    #     axs[1,1].set_title(f"Predictive uncertainty")

    #     plt.subplots_adjust(wspace=0)
    #     fig.canvas.draw()
    #     fig.canvas.flush_events()
    #     plt.pause(1e-9)
    #     cb.remove()
    # plt.show()


if __name__ == '__main__':
    main()
