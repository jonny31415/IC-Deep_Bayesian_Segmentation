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
    def __init__(self, n_classes, kernel_regularizer=None, **kwargs):
        super(BasicSegmentationHead, self).__init__(**kwargs)

        self.n_classes = n_classes 
        self.kernel_regularizer = kernel_regularizer

        if self.kernel_regularizer == 'l1':
            self.kernel_regularizer = tf.keras.regularizers.L1(l1=0.01, **kwargs)
        elif self.kernel_regularizer == 'l2':
            self.kernel_regularizer = tf.keras.regularizers.L2(l2=0.01, **kwargs)
        else:
            self.kernel_regularizer = None

        self.conv1 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_regularizer=self.kernel_regularizer)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation('relu')   

        self.conv2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same', kernel_regularizer=self.kernel_regularizer)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.Activation('relu')          

        self.conv3 = tf.keras.layers.Conv2D(self.n_classes, kernel_size=1, strides=1, padding='same', kernel_regularizer=self.kernel_regularizer)

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
        config = super(BasicSegmentationHead, self).get_config()
        config.update({"num_classes": self.n_classes, "kernel_regularizer": self.kernel_regularizer})
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


class HRNetOCR(tf.keras.Model):
    def __init__(self, n_classes, filters=64, height=None, width=None, final_activation="softmax",
                 spatial_ocr_scale=1, spatial_context_scale=1, **kwargs):
        super(HRNetOCR, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.filters = filters
        self.height = height
        self.width = width
        self.final_activation = final_activation
        self.spatial_ocr_scale = spatial_ocr_scale
        self.spatial_context_scale = spatial_context_scale

        axis = 3 if K.image_data_format() == "channels_last" else 1

        # Stem Net
        self.conv3x3_bn_relu_1 = ConvolutionBnActivation(filters, (3, 3))
        self.conv3x3_bn_relu_2 = ConvolutionBnActivation(filters, (3, 3))

        # stage 1
        self.bottleneck_downsample = BottleneckBlock(64, downsample=True)
        self.bottleneck_1 = BottleneckBlock(64)
        self.bottleneck_2 = BottleneckBlock(64)
        self.bottleneck_3 = BottleneckBlock(64)

        # Stage 2
        # Transition
        self.conv3x3_bn_relu_stage2_1 = ConvolutionBnActivation(48, (3, 3), momentum=0.1)
        self.conv3x3_bn_relu_stage2_2 = ConvolutionBnActivation(96, (3, 3), strides=(2, 2), momentum=0.1)

        # Stage
        # num_modules=1, num_branches=2, blocks=[4, 4], channels=[48, 96]
        self.hrn_stage2_module_1 = HighResolutionModule(num_branches=2, blocks=[4, 4], filters=[48, 96])
        self.hrn_stage2_module_2 = HighResolutionModule(num_branches=2, blocks=[4, 4], filters=[48, 96])

        # Stage 3
        # Transition
        self.conv3x3_bn_relu_stage3 = ConvolutionBnActivation(192, (3, 3), strides=(2, 2), momentum=0.1)

        # Stage
        # num_modules=4, num_branches=3, blocks=[4, 4, 4], channels=[48, 96, 192]
        self.hrn_stage3_module_1 = HighResolutionModule(num_branches=3, blocks=[4, 4, 4], filters=[48, 96, 192])
        self.hrn_stage3_module_2 = HighResolutionModule(num_branches=3, blocks=[4, 4, 4], filters=[48, 96, 192])
        self.hrn_stage3_module_3 = HighResolutionModule(num_branches=3, blocks=[4, 4, 4], filters=[48, 96, 192])
        self.hrn_stage3_module_4 = HighResolutionModule(num_branches=3, blocks=[4, 4, 4], filters=[48, 96, 192])

        # Stage 4
        # Transition
        self.conv3x3_bn_relu_stage4 = ConvolutionBnActivation(384, (3, 3), strides=(2, 2), momentum=0.1)

        # Stage
        # num_modules=3, num_branches=4, num_blocks=[4, 4, 4, 4], num_channels=[48, 96, 192, 384]
        self.hrn_stage4_module_1 = HighResolutionModule(num_branches=4, blocks=[4, 4, 4, 4], filters=[48, 96, 192, 384])
        self.hrn_stage4_module_2 = HighResolutionModule(num_branches=4, blocks=[4, 4, 4, 4], filters=[48, 96, 192, 384])
        self.hrn_stage4_module_3 = HighResolutionModule(num_branches=4, blocks=[4, 4, 4, 4], filters=[48, 96, 192, 384])

        # Upsampling and Concatentation of stages
        self.upsample_x2 = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")
        self.upsample_x4 = tf.keras.layers.UpSampling2D(size=4, interpolation="bilinear")
        self.upsample_x8 = tf.keras.layers.UpSampling2D(size=8, interpolation="bilinear")

        self.concat = tf.keras.layers.Concatenate(axis=axis)

        # OCR
        self.aux_head = tf.keras.Sequential([
            ConvolutionBnActivation(720, (1, 1)),
            tf.keras.layers.Conv2D(filters=self.n_classes, kernel_size=(1, 1), use_bias=True),
            tf.keras.layers.Activation(final_activation)
        ])
        self.conv3x3_bn_relu_ocr = ConvolutionBnActivation(512, (3, 3))

        self.spatial_context = SpatialGatherModule(scale=spatial_context_scale)
        self.spatial_ocr = SpatialOCRModule(512, scale=spatial_ocr_scale, dropout=0.05)

        self.final_conv3x3 = tf.keras.layers.Conv2D(filters=self.n_classes, kernel_size=(1, 1), use_bias=True)
        self.final_activation = tf.keras.layers.Activation(final_activation)

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = True

        x = self.conv3x3_bn_relu_1(inputs, training=training)
        x = self.conv3x3_bn_relu_2(x, training=training)

        # Stage 1
        x = self.bottleneck_downsample(x, training=training)
        x = self.bottleneck_1(x, training=training)
        x = self.bottleneck_2(x, training=training)
        x = self.bottleneck_3(x, training=training)

        # Stage 2
        x_1 = self.conv3x3_bn_relu_stage2_1(x, training=training)
        x_2 = self.conv3x3_bn_relu_stage2_2(x, training=training)  # includes strided convolution

        y_list = self.hrn_stage2_module_1(x_1, x_2, None, None, training=training)
        y_list = self.hrn_stage2_module_2(y_list[0], y_list[1], None, None, training=training)

        # Stage 3
        x_3 = self.conv3x3_bn_relu_stage3(y_list[1], training=training)  # includes strided convolution

        y_list = self.hrn_stage3_module_1(y_list[0], y_list[1], x_3, None, training=training)
        y_list = self.hrn_stage3_module_2(y_list[0], y_list[1], y_list[2], None, training=training)
        y_list = self.hrn_stage3_module_3(y_list[0], y_list[1], y_list[2], None, training=training)
        y_list = self.hrn_stage3_module_4(y_list[0], y_list[1], y_list[2], None, training=training)

        # Stage 4
        x_4 = self.conv3x3_bn_relu_stage4(y_list[2], training=training)

        y_list = self.hrn_stage4_module_1(y_list[0], y_list[1], y_list[2], x_4, training=training)
        y_list = self.hrn_stage4_module_2(y_list[0], y_list[1], y_list[2], y_list[3], training=training)
        y_list = self.hrn_stage4_module_3(y_list[0], y_list[1], y_list[2], y_list[3], training=training)

        # Upsampling + Concatentation
        x_2 = self.upsample_x2(y_list[1])
        x_3 = self.upsample_x4(y_list[2])
        x_4 = self.upsample_x8(y_list[3])

        feats = self.concat([y_list[0], x_2, x_3, x_4])

        # OCR
        aux = self.aux_head(feats)

        feats = self.conv3x3_bn_relu_ocr(feats)

        context = self.spatial_context(feats, aux)
        feats = self.spatial_ocr(feats, context)

        out = self.final_conv3x3(feats)
        out = self.final_activation(out)

        return out

    def model(self):
        x = tf.keras.layers.Input(shape=(self.height, self.width, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class HRNet(tf.keras.Model):
    def __init__(self, n_classes, filters=64, height=None, width=None, final_activation="softmax",
                 spatial_ocr_scale=1, spatial_context_scale=1, **kwargs):
        super(HRNet, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.filters = filters
        self.height = height
        self.width = width
        self.final_activation = final_activation
        self.spatial_ocr_scale = spatial_ocr_scale
        self.spatial_context_scale = spatial_context_scale

        axis = 3 if K.image_data_format() == "channels_last" else 1

        # Stem Net
        self.conv3x3_bn_relu_1 = ConvolutionBnActivation(filters, (3, 3))
        self.conv3x3_bn_relu_2 = ConvolutionBnActivation(filters, (3, 3))

        # stage 1
        self.bottleneck_downsample = BottleneckBlock(64, downsample=True)
        self.bottleneck_1 = BottleneckBlock(64)
        self.bottleneck_2 = BottleneckBlock(64)
        self.bottleneck_3 = BottleneckBlock(64)

        # Stage 2
        # Transition
        self.conv3x3_bn_relu_stage2_1 = ConvolutionBnActivation(48, (3, 3), momentum=0.1)
        self.conv3x3_bn_relu_stage2_2 = ConvolutionBnActivation(96, (3, 3), strides=(2, 2), momentum=0.1)

        # Stage
        # num_modules=1, num_branches=2, blocks=[4, 4], channels=[48, 96]
        self.hrn_stage2_module_1 = HighResolutionModule(num_branches=2, blocks=[4, 4], filters=[48, 96])
        self.hrn_stage2_module_2 = HighResolutionModule(num_branches=2, blocks=[4, 4], filters=[48, 96])

        # Stage 3
        # Transition
        self.conv3x3_bn_relu_stage3 = ConvolutionBnActivation(192, (3, 3), strides=(2, 2), momentum=0.1)

        # Stage
        # num_modules=4, num_branches=3, blocks=[4, 4, 4], channels=[48, 96, 192]
        self.hrn_stage3_module_1 = HighResolutionModule(num_branches=3, blocks=[4, 4, 4], filters=[48, 96, 192])
        self.hrn_stage3_module_2 = HighResolutionModule(num_branches=3, blocks=[4, 4, 4], filters=[48, 96, 192])
        self.hrn_stage3_module_3 = HighResolutionModule(num_branches=3, blocks=[4, 4, 4], filters=[48, 96, 192])
        self.hrn_stage3_module_4 = HighResolutionModule(num_branches=3, blocks=[4, 4, 4], filters=[48, 96, 192])

        # Stage 4
        # Transition
        self.conv3x3_bn_relu_stage4 = ConvolutionBnActivation(384, (3, 3), strides=(2, 2), momentum=0.1)

        # Stage
        # num_modules=3, num_branches=4, num_blocks=[4, 4, 4, 4], num_channels=[48, 96, 192, 384]
        self.hrn_stage4_module_1 = HighResolutionModule(num_branches=4, blocks=[4, 4, 4, 4], filters=[48, 96, 192, 384])
        self.hrn_stage4_module_2 = HighResolutionModule(num_branches=4, blocks=[4, 4, 4, 4], filters=[48, 96, 192, 384])
        self.hrn_stage4_module_3 = HighResolutionModule(num_branches=4, blocks=[4, 4, 4, 4], filters=[48, 96, 192, 384])

        # Upsampling and Concatentation of stages
        self.upsample_x2 = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")
        self.upsample_x4 = tf.keras.layers.UpSampling2D(size=4, interpolation="bilinear")
        self.upsample_x8 = tf.keras.layers.UpSampling2D(size=8, interpolation="bilinear")

        self.concat = tf.keras.layers.Concatenate(axis=axis)

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = True

        x = self.conv3x3_bn_relu_1(inputs, training=training)
        x = self.conv3x3_bn_relu_2(x, training=training)

        # Stage 1
        x = self.bottleneck_downsample(x, training=training)
        x = self.bottleneck_1(x, training=training)
        x = self.bottleneck_2(x, training=training)
        x = self.bottleneck_3(x, training=training)

        # Stage 2
        x_1 = self.conv3x3_bn_relu_stage2_1(x, training=training)
        x_2 = self.conv3x3_bn_relu_stage2_2(x, training=training)  # includes strided convolution

        y_list = self.hrn_stage2_module_1(x_1, x_2, None, None, training=training)
        y_list = self.hrn_stage2_module_2(y_list[0], y_list[1], None, None, training=training)

        # Stage 3
        x_3 = self.conv3x3_bn_relu_stage3(y_list[1], training=training)  # includes strided convolution

        y_list = self.hrn_stage3_module_1(y_list[0], y_list[1], x_3, None, training=training)
        y_list = self.hrn_stage3_module_2(y_list[0], y_list[1], y_list[2], None, training=training)
        y_list = self.hrn_stage3_module_3(y_list[0], y_list[1], y_list[2], None, training=training)
        y_list = self.hrn_stage3_module_4(y_list[0], y_list[1], y_list[2], None, training=training)

        # Stage 4
        x_4 = self.conv3x3_bn_relu_stage4(y_list[2], training=training)

        y_list = self.hrn_stage4_module_1(y_list[0], y_list[1], y_list[2], x_4, training=training)
        y_list = self.hrn_stage4_module_2(y_list[0], y_list[1], y_list[2], y_list[3], training=training)
        y_list = self.hrn_stage4_module_3(y_list[0], y_list[1], y_list[2], y_list[3], training=training)

        # Upsampling + Concatentation
        x_2 = self.upsample_x2(y_list[1])
        x_3 = self.upsample_x4(y_list[2])
        x_4 = self.upsample_x8(y_list[3])

        feats = self.concat([y_list[0], x_2, x_3, x_4])

        return feats

    def model(self):
        x = tf.keras.layers.Input(shape=(self.height, self.width, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class OCR(tf.keras.Model):
    def __init__(self, n_classes, filters=64, height=None, width=None, final_activation="softmax",
                 spatial_ocr_scale=1, spatial_context_scale=1, **kwargs):
        super(OCR, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.filters = filters
        self.height = height
        self.width = width
        self.final_activation = final_activation
        self.spatial_ocr_scale = spatial_ocr_scale
        self.spatial_context_scale = spatial_context_scale

        # OCR
        self.aux_head = tf.keras.Sequential([
            ConvolutionBnActivation(720, (1, 1)),
            tf.keras.layers.Conv2D(filters=self.n_classes, kernel_size=(1, 1), use_bias=True),
            tf.keras.layers.Activation(final_activation)
        ])
        self.conv3x3_bn_relu_ocr = ConvolutionBnActivation(512, (3, 3))

        self.spatial_context = SpatialGatherModule(scale=spatial_context_scale)
        self.spatial_ocr = SpatialOCRModule(512, scale=spatial_ocr_scale, dropout=0.05)

        self.final_conv3x3 = tf.keras.layers.Conv2D(filters=self.n_classes, kernel_size=(1, 1), use_bias=True)
        self.final_activation = tf.keras.layers.Activation(final_activation)

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = True

        feats = inputs
        aux = self.aux_head(feats)

        feats = self.conv3x3_bn_relu_ocr(feats)

        context = self.spatial_context(feats, aux)
        feats = self.spatial_ocr(feats, context)

        out = self.final_conv3x3(feats)
        out = self.final_activation(out)

        return out

    def model(self):
        x = tf.keras.layers.Input(shape=(self.height, self.width, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
