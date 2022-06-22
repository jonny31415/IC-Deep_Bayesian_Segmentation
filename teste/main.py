#!/usr/bin/env python3

import os
from datetime import datetime
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from models.custom_models import BasicSegmentationHead, BayesianSegmentationHeadAleatoric, BayesianSegmentationHeadEpistemic, BayesianSegmentationHeadFullProb
from metrics.onehot_miou import OneHotMeanIoU
from datasets.cityscapes import CityscapesGenerator
from utils import print_np, get_layers, get_summary
from callbacks.tensorboard import TensorboardCallback
from losses.NLL import NLL
from trunks.ResNet import get_resnet

# Disable memory growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_memory_growth(physical_devices[1], True)

############
## PARAMS ##
############
ONE_IMG_PATH = os.path.join('/media', 'olorin', 'Documentos', 'datasets', 'cityscapes', 'leftImg8bit', 'train', 'aachen')
ONE_LABEL_PATH = os.path.join('/media', 'olorin', 'Documentos', 'datasets', 'cityscapes', 'gtFine', 'train', 'aachen')

TRAIN_INPUT_DIR = os.path.join('/media', 'olorin', 'Documentos', 'datasets', 'cityscapes', 'leftImg8bit', 'train')
TRAIN_TARGET_DIR = os.path.join('/media', 'olorin', 'Documentos', 'datasets', 'cityscapes', 'gtFine', 'train')
VAL_INPUT_DIR = os.path.join('/media', 'olorin', 'Documentos', 'datasets', 'cityscapes', 'leftImg8bit', 'val')
VAL_TARGET_DIR = os.path.join('/media', 'olorin', 'Documentos', 'datasets', 'cityscapes', 'gtFine', 'val')
BATCH_SIZE = 1 # Max_BATCH_SIZE for 256x512: 40 -> rede mais simples # Max_BATCH_SIZE for 1024x2048: 2 -> rede mais simples
N_CLASSES = 19

HEIGHT = 512
WIDTH = 1024
CHANNELS = 3
IMG_SIZE = (HEIGHT, WIDTH)
PRETRAINED = True
DATA_AUG = True
MAX_EPOCHS = 200
POLY_EXP = 2
TRUNK_NAME = 'ResNet50'
MODEL_NAME = 'BasicSegHead'
KERNEL_REGULARIZER = None

def get_trunk(input_shape, selected_trunk):
    if selected_trunk == 'ResNet50':
        trunk = get_resnet(input_shape, selected_trunk, output_stride=None)
    else:  
        raise NotImplementedError
 
    return trunk, trunk.output


def get_model(input_shape, train_gen_len, selected_model, selected_trunk):
    trunk, trunk_features = get_trunk(input_shape, selected_trunk)

    if selected_model == 'BasicSegHead':
        segmentation_head = BasicSegmentationHead(N_CLASSES, kernel_regularizer=KERNEL_REGULARIZER)
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


def unnormalize(img, mean, std):
    unnorm_img = np.zeros(img.shape)
    unnorm_img[:, :, :, 0] = img[:, :, :, 0]*std[0] + mean[0]
    unnorm_img[:, :, :, 1] = img[:, :, :, 1]*std[1] + mean[1]
    unnorm_img[:, :, :, 2] = img[:, :, :, 2]*std[2] + mean[2]

    return (unnorm_img*255).astype(np.uint8)

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

    one_img_test = sorted(
        [
            os.path.join(ONE_IMG_PATH, fname)
            for fname in os.listdir(os.path.join(ONE_IMG_PATH))
                if fname.endswith(".png")
        ]
    )

    one_label_test = sorted(
        [
            os.path.join(ONE_LABEL_PATH, fname)
            for fname in os.listdir(os.path.join(ONE_LABEL_PATH))
                if fname.endswith("gtFine_labelIds.png") and not fname.startswith(".")
        ]
    )

    one_img_test = one_img_test[0:1]
    one_label_test = one_label_test[0:1]

    train_one_img_gen = CityscapesGenerator(BATCH_SIZE, IMG_SIZE, one_img_test, one_label_test, data_aug=DATA_AUG)
    val_one_img_gen = CityscapesGenerator(BATCH_SIZE, IMG_SIZE, one_img_test, one_label_test)

    # Creates train dataset
    train_gen = CityscapesGenerator(BATCH_SIZE, IMG_SIZE, train_input_img_paths, train_target_img_paths, data_aug=DATA_AUG)

    # Creates validation dataset
    val_gen = CityscapesGenerator(BATCH_SIZE, IMG_SIZE, val_input_img_paths, val_target_img_paths)

    train_dataset = train_gen
    validation_dataset = val_gen

    model, loss_fn = get_model(input_shape, train_gen_len = len(train_dataset), selected_model=MODEL_NAME, selected_trunk=TRUNK_NAME)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9), loss=loss_fn, 
                    metrics=['accuracy', OneHotMeanIoU(N_CLASSES)])
    # get_summary(model)
    # input('aqui')

    if PRETRAINED:
        model.load_weights("weights/ResNet50-BasicSegHead-2022-06-20 13:50:29.377726/092-0.18.h5", by_name=True)

    # Learning rate decay schedule
    exp_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=5,
    decay_rate=0.9)

    def poly_lr_schedule(epoch, initial_learning_rate=1e-2):
        return initial_learning_rate*tf.pow(1 - epoch / MAX_EPOCHS, POLY_EXP)
    
    lr_schedule = exp_lr_schedule

    datenow = str(datetime.now())
    # params_dict = {"Optimizer": 'SGD', "Loss": loss_fn.__name__, "Arch": MODEL_NAME, "Trunk": TRUNK_NAME, 
    #                "Batch size": BATCH_SIZE}
    params_dict = {"Optimizer": 'SGD', "Loss": model.loss.name, "Arch": MODEL_NAME, "Trunk": TRUNK_NAME, 
                   "Batch size": BATCH_SIZE}
    
    my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_one_hot_mean_io_u", patience=20, mode="max"),
    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('./weights/', TRUNK_NAME+'-'+MODEL_NAME+'-'+ datenow, 
                                        '{epoch:03d}-{val_one_hot_mean_io_u:.2f}.h5'), monitor="val_one_hot_mean_io_u", 
                                        mode="max", save_best_only=True, verbose=True),
    tf.keras.callbacks.TensorBoard(log_dir=f'./logs/{TRUNK_NAME}-{MODEL_NAME}'+'-'+ datenow, write_images=True),
    TensorboardCallback(logdir=f'./logs/{TRUNK_NAME}-{MODEL_NAME}'+'-'+ datenow +'/Images/', val_data=validation_dataset, 
                        monitor="val_one_hot_mean_io_u", hparams = params_dict),
    tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=True),
    hp.KerasCallback(writer=f'./logs/{TRUNK_NAME}-{MODEL_NAME}'+'-'+ datenow, hparams=params_dict)
    ]

    # my_callbacks = [
    # tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min"),
    # tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('./weights/', TRUNK_NAME+'-'+MODEL_NAME+'-'+ datenow, 
    #                                     '{epoch:03d}-{val_one_hot_mean_io_u:.2f}.h5'), monitor="val_loss", 
    #                                     mode="min", save_best_only=True, verbose=True),
    # tf.keras.callbacks.TensorBoard(log_dir=f'./logs/{TRUNK_NAME}-{MODEL_NAME}'+'-'+ datenow, write_images=True),
    # TensorboardCallback(logdir=f'./logs/{TRUNK_NAME}-{MODEL_NAME}'+'-'+ datenow +'/Images/', val_data=validation_dataset, 
    #                     monitor="val_one_hot_mean_io_u", hparams = params_dict),
    # tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=True),
    # hp.KerasCallback(writer=f'./logs/{TRUNK_NAME}-{MODEL_NAME}'+'-'+ datenow, hparams=params_dict)
    # ]

    model.fit(x=train_dataset, epochs=MAX_EPOCHS, verbose=1, callbacks=my_callbacks, 
            validation_data=validation_dataset, shuffle=True)

    # model.fit(x=train_dataset, epochs=MAX_EPOCHS, verbose=1, callbacks=my_callbacks, shuffle=True)
    
if __name__ == '__main__':
    main()