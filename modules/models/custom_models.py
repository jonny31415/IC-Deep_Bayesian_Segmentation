#!/usr/bin/env python3

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
from .custom_layers import ConvolutionBnActivation, BottleneckBlock, HighResolutionModule, \
    SpatialGatherModule, SpatialOCRModule
from utils import print_np
from losses.NLL import kl_approx

tfd = tfp.distributions
tfpl = tfp.layers


class BasicSegmentationHead(tf.keras.Model):
    def __init__(self, n_classes, input_shape, output_shape, **kwargs):
        super(BasicSegmentationHead, self).__init__(**kwargs)

        if len(input_shape) != 4:
            raise ValueError(f"Input shape must be of rank 4, but input_shape was of rank {len(input_shape)}")
        if len(output_shape) != 4:
            raise ValueError(f"Output shape must be of rank 4, but output_shape was of rank {len(output_shape)}")

        self.n_classes = n_classes 
        self.upsample_factor = output_shape[1]//input_shape[1]

        self.conv1 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation('relu')   

        self.conv2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.Activation('relu')          

        self.conv3 = tf.keras.layers.Conv2D(self.n_classes, kernel_size=1, strides=1, padding='same')

        self.upsample_layer = tf.keras.layers.UpSampling2D(size=self.upsample_factor, interpolation="bilinear")

    def call(self, inputs):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.upsample_layer(x)

        return x

    def get_config(self):
        config = super(BasicSegmentationHead, self).get_config()
        config.update({"num_classes": self.n_classes})
        return config

class BayesianSegmentationHeadAleatoric(tf.keras.Model):
    def __init__(self, n_classes, train_gen_len, use_exact_kl=False, **kwargs):
        super(BayesianSegmentationHeadAleatoric, self).__init__(**kwargs)

        self.n_classes = n_classes 
        self.use_exact_kl = use_exact_kl
        
        if self.use_exact_kl:
            self.divergence_fn = lambda q,p,_ : tfd.kl_divergence(q,p) / train_gen_len
        else:
            self.divergence_fn = lambda q, p, q_tensor : kl_approx(q, p, q_tensor) / train_gen_len

        self.conv1 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation('relu')   

        self.conv2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.Activation('relu')          

        self.conv3 = tf.keras.layers.Conv2D(tfpl.OneHotCategorical.params_size(self.n_classes), kernel_size=1, strides=1, padding='same')
        
        self.upsample_layer = tf.keras.layers.UpSampling2D(size=32, interpolation="bilinear")

        self.one_hot_cat = tfpl.OneHotCategorical(self.n_classes)

    def call(self, inputs, training=None):
        if training is None:
            training = True

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.upsample_layer(x)
        x = self.one_hot_cat(x)

        return x

    def get_config(self):
        config = super(BayesianSegmentationHeadAleatoric, self).get_config()
        config.update({"num_classes": self.n_classes, "use_exact_kl": self.use_exact_kl})
        return config


class BayesianSegmentationHeadEpistemic(tf.keras.Model):
    def __init__(self, n_classes, train_gen_len, use_exact_kl=False, **kwargs):
        super(BayesianSegmentationHeadEpistemic, self).__init__(**kwargs)

        self.n_classes = n_classes 
        self.use_exact_kl = use_exact_kl
        
        if self.use_exact_kl:
            self.divergence_fn = lambda q,p,_ : tfd.kl_divergence(q,p) / train_gen_len
        else:
            self.divergence_fn = lambda q, p, q_tensor : kl_approx(q, p, q_tensor) / train_gen_len

        self.conv1 = tfp.layers.Convolution2DReparameterization(
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_posterior_fn=tfpl.default_multivariate_normal_fn,
            kernel_prior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_divergence_fn=self.divergence_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_divergence_fn=self.divergence_fn,
            **kwargs
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation('relu')   

        self.conv2 = tfp.layers.Convolution2DReparameterization(
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_posterior_fn=tfpl.default_multivariate_normal_fn,
            kernel_prior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_divergence_fn=self.divergence_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_divergence_fn=self.divergence_fn,
            **kwargs
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.Activation('relu')          

        self.conv3 = tfp.layers.Convolution2DReparameterization(
            filters=self.n_classes,
            kernel_size=1,
            strides=1,
            padding='same',
            kernel_posterior_fn=tfpl.default_multivariate_normal_fn,
            kernel_prior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_divergence_fn=self.divergence_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_divergence_fn=self.divergence_fn,
            **kwargs
        )

        self.upsample_layer = tf.keras.layers.UpSampling2D(size=32, interpolation="bilinear")

    def call(self, inputs, training=None):
        if training is None:
            training = True

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.upsample_layer(x)

        return x

    def get_config(self):
        config = super(BayesianSegmentationHeadEpistemic, self).get_config()
        config.update({"num_classes": self.n_classes, "use_exact_kl": self.use_exact_kl})
        return config


class BayesianSegmentationHeadFullProbOneConv(tf.keras.Model):
    def __init__(self, n_classes, train_gen_len, use_exact_kl=False, **kwargs):
        super(BayesianSegmentationHeadFullProbOneConv, self).__init__(**kwargs)

        self.n_classes = n_classes 
        self.use_exact_kl = use_exact_kl
        
        if self.use_exact_kl:
            self.divergence_fn = lambda q,p,_ : tfd.kl_divergence(q,p) / train_gen_len
        else:
            self.divergence_fn = lambda q, p, q_tensor : kl_approx(q, p, q_tensor) / train_gen_len

        self.conv1 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation('relu')   

        self.conv2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.Activation('relu')          

        self.conv3 = self.conv3 = tfp.layers.Convolution2DReparameterization(
            filters=tfpl.OneHotCategorical.params_size(self.n_classes),
            kernel_size=1,
            strides=1,
            padding='same',
            kernel_posterior_fn=tfpl.default_multivariate_normal_fn,
            kernel_prior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_divergence_fn=self.divergence_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_divergence_fn=self.divergence_fn,
            **kwargs
        )

        self.upsample_layer = tf.keras.layers.UpSampling2D(size=32, interpolation="bilinear")

        self.one_hot_cat = tfpl.OneHotCategorical(self.n_classes)

    def call(self, inputs, training=None):
        if training is None:
            training = True

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.upsample_layer(x)
        x = self.one_hot_cat(x)

        return x

    def get_config(self):
        config = super(BayesianSegmentationHeadFullProb, self).get_config()
        config.update({"num_classes": self.n_classes, "use_exact_kl": self.use_exact_kl})
        return config


class BayesianSegmentationHeadFullProb(tf.keras.Model):
    def __init__(self, n_classes, train_gen_len, use_exact_kl=False, **kwargs):
        super(BayesianSegmentationHeadFullProb, self).__init__(**kwargs)

        self.n_classes = n_classes 
        self.use_exact_kl = use_exact_kl
        
        if self.use_exact_kl:
            self.divergence_fn = lambda q,p,_ : tfd.kl_divergence(q,p) / train_gen_len
        else:
            self.divergence_fn = lambda q, p, q_tensor : kl_approx(q, p, q_tensor) / train_gen_len

        self.conv1 = tfp.layers.Convolution2DReparameterization(
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_posterior_fn=tfpl.default_multivariate_normal_fn,
            kernel_prior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_divergence_fn=self.divergence_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_divergence_fn=self.divergence_fn,
            **kwargs
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation('relu')   

        self.conv2 = tfp.layers.Convolution2DReparameterization(
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same',
            kernel_posterior_fn=tfpl.default_multivariate_normal_fn,
            kernel_prior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_divergence_fn=self.divergence_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_divergence_fn=self.divergence_fn,
            **kwargs
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.Activation('relu')          

        self.conv3 = tfp.layers.Convolution2DReparameterization(
            filters=tfpl.OneHotCategorical.params_size(self.n_classes),
            kernel_size=1,
            strides=1,
            padding='same',
            kernel_posterior_fn=tfpl.default_multivariate_normal_fn,
            kernel_prior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_divergence_fn=self.divergence_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_divergence_fn=self.divergence_fn,
            **kwargs
        )

        self.upsample_layer = tf.keras.layers.UpSampling2D(size=32, interpolation="bilinear")

        self.one_hot_cat = tfpl.OneHotCategorical(self.n_classes)

    def call(self, inputs, training=None):
        if training is None:
            training = True

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.upsample_layer(x)
        x = self.one_hot_cat(x)

        return x

    def get_config(self):
        config = super(BayesianSegmentationHeadFullProb, self).get_config()
        config.update({"num_classes": self.n_classes, "use_exact_kl": self.use_exact_kl})
        return config


# Ver o no. de filtros
class AuxSegmentationHead(tf.keras.Model):
    def __init__(self, n_classes, filters=64, height=None, width=None, **kwargs):
        super(AuxSegmentationHead, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.filters = filters
        self.height = height
        self.width = width

        self.conv1x1_bn_relu = ConvolutionBnActivation(self.filters, (1, 1), post_activation='relu')
        self.conv1x1 = tf.keras.layers.Conv2D(self.n_classes, (1, 1), strides=(1, 1), padding='same')

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = True

        x = self.conv1x1_bn_relu(inputs, training=training)
        x = self.conv1x1(x, training=training)

        return x


# Ver o no. de filtros
class AttentionHead(tf.keras.Model):
    def __init__(self, filters=128, height=None, width=None, **kwargs):
        super(AttentionHead, self).__init__(**kwargs)

        self.filters = filters
        self.height = height
        self.width = width

        self.conv3x3_bn_relu_1 = ConvolutionBnActivation(self.filters, (3, 3), post_activation='relu')
        self.conv3x3_bn_relu_2 = ConvolutionBnActivation(self.filters/8, (3, 3), post_activation='relu')
        self.conv1x1 = tf.keras.layers.Conv2D(1, (1, 1), strides=(1, 1), padding='same')

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = True

        x = self.conv3x3_bn_relu_1(inputs, training=training)
        x = self.conv3x3_bn_relu_2(x, training=training)
        x = self.conv1x1(x, training=training)

        return x