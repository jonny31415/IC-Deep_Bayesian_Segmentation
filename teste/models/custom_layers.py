#!/usr/bin/env python3

import tensorflow as tf
import tensorflow.keras.backend as K


class ConvolutionBnActivation(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides=(1, 1), padding="same", data_format=None, dilation_rate=(1, 1),
                 groups=1, activation=None, kernel_initializer="glorot_uniform", bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                 use_batchnorm=False,
                 axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, trainable=True,
                 post_activation="relu", block_name=None):
        super(ConvolutionBnActivation, self).__init__()

        # 2D Convolution Arguments
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = not use_batchnorm
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        # Batch Normalization Arguments
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.trainable = trainable

        self.block_name = block_name

        self.conv = None
        self.bn = None
        self.post_activation = tf.keras.layers.Activation(post_activation)

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            name=self.block_name + "_conv" if self.block_name is not None else None

        )

        self.bn = tf.keras.layers.BatchNormalization(
            axis=self.axis,
            momentum=self.momentum,
            epsilon=self.epsilon,
            center=self.center,
            scale=self.scale,
            trainable=self.trainable,
            name=self.block_name + "_bn" if self.block_name is not None else None
        )

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.post_activation(x)

        return x

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return [input_shape[0], input_shape[1], input_shape[2], self.filters]


class BottleneckBlock(tf.keras.layers.Layer):
    def __init__(self, filters, downsample=False, expansion=4):
        super(BottleneckBlock, self).__init__()

        self.ds = downsample

        self.conv3x3_bn_relu_1 = ConvolutionBnActivation(filters, (1, 1), momentum=0.1)
        self.conv3x3_bn_relu_2 = ConvolutionBnActivation(filters, (3, 3), momentum=0.1)
        self.conv3x3_bn = ConvolutionBnActivation(filters * expansion, (1, 1), momentum=0.1, post_activation="linear")

        if downsample:
            self.downsample = ConvolutionBnActivation(filters * expansion, (1, 1), momentum=0.1)

        self.relu = tf.keras.layers.Activation("relu")

    def call(self, input, training=None):
        residual = input

        out = self.conv3x3_bn_relu_1(input, training=training)
        out = self.conv3x3_bn_relu_2(out, training=training)
        out = self.conv3x3_bn(out, training=training)

        if self.ds:
            residual = self.downsample(input)

        out = out + residual
        out = self.relu(out)

        return out


# Basic Block for HRNetOCR
class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(BasicBlock, self).__init__()

        self.conv3x3_bn_relu = ConvolutionBnActivation(filters, (3, 3), momentum=0.1)
        self.conv3x3_bn = ConvolutionBnActivation(filters, (3, 3), momentum=0.1, post_activation="linear")

        self.relu = tf.keras.layers.Activation("relu")

    def call(self, input, training=None):
        residual = input

        out = self.conv3x3_bn_relu(input, training=training)
        out = self.conv3x3_bn(out, training=training)

        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(tf.keras.layers.Layer):
    def __init__(self, num_branches, blocks, filters):
        # filters_in unnecessary since it equals filters
        super(HighResolutionModule, self).__init__()

        self.num_branches = num_branches
        self.filters = filters
        self.num_in_channels = filters[0]

        self._check_branches(num_branches, blocks, filters)

        # Make Branches
        self.branch_1 = tf.keras.Sequential(
            [BasicBlock(filters[0]), BasicBlock(filters[0]), BasicBlock(filters[0]), BasicBlock(filters[0])])
        self.branch_2 = tf.keras.Sequential(
            [BasicBlock(filters[1]), BasicBlock(filters[1]), BasicBlock(filters[1]), BasicBlock(filters[1])])
        self.branch_3 = tf.keras.Sequential([BasicBlock(filters[2]), BasicBlock(filters[2]), BasicBlock(filters[2]),
                                             BasicBlock(filters[2])]) if num_branches >= 3 else None
        self.branch_4 = tf.keras.Sequential([BasicBlock(filters[3]), BasicBlock(filters[3]), BasicBlock(filters[3]),
                                             BasicBlock(filters[3])]) if num_branches >= 4 else None

        self.fuse_layers = self._make_fuse_layers()
        self.relu = tf.keras.layers.Activation("relu")

    def _check_branches(self, num_branches, blocks, filters):
        if num_branches != len(blocks):
            raise ValueError(
                "'num_branches' = {} is not equal to length of 'blocks' = {}".format(num_branches, len(blocks)))

        if num_branches != len(filters):
            raise ValueError(
                "'num_branches' = {} is not equal to length of 'filters' = {}".format(num_branches, len(filters)))

    def _make_fuse_layers(self):
        fuse_layers = []
        for i in range(self.num_branches):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    fuse_layer.append(
                        ConvolutionBnActivation(self.filters[i], (1, 1), momentum=0.1, post_activation="linear"))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv3x3s.append(
                                ConvolutionBnActivation(self.filters[i], (3, 3), strides=(2, 2), momentum=0.1,
                                                        post_activation="linear"))
                        else:
                            conv3x3s.append(
                                ConvolutionBnActivation(self.filters[j], (3, 3), strides=(2, 2), momentum=0.1))
                    fuse_layer.append(tf.keras.Sequential(conv3x3s))

            fuse_layers.append(fuse_layer)

        return fuse_layers

    def call(self, input1, input2, input3, input4, training=None):
        x_1 = self.branch_1(input1, training=training)
        x_2 = self.branch_2(input2, training=training)
        x_3 = self.branch_3(input3, training=training) if self.num_branches >= 3 else None
        x_4 = self.branch_4(input4, training=training) if self.num_branches >= 4 else None

        x = [x_1, x_2]
        if x_3 is not None:
            x = [x_1, x_2, x_3]
        if x_4 is not None:
            x = [x_1, x_2, x_3, x_4]

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y += x[j]
                elif j > i:
                    f = self.fuse_layers[i][j](x[j])
                    scale_factor = int(x[i].shape[-2] / f.shape[-2])
                    if scale_factor > 1:
                        y += tf.keras.layers.UpSampling2D(size=scale_factor, interpolation="bilinear")(f)
                    else:
                        y += f
                else:
                    y += self.fuse_layers[i][j](x[j])

            x_fuse.append(self.relu(y))

        return x_fuse


class SpatialGatherModule(tf.keras.layers.Layer):
    def __init__(self, scale=1):
        super(SpatialGatherModule, self).__init__()

        self.scale = scale

        self.softmax = tf.keras.layers.Activation("softmax")

    def call(self, features, probabilities, training=None):
        if K.image_data_format() == "channels_last":
            BS, H, W, C = probabilities.shape
            p = tf.keras.layers.Reshape((-1, C))(probabilities)  # (BS, N, C)
            f = tf.keras.layers.Reshape((-1, features.shape[-1]))(features)  # (BS, N, C2)

            p = self.softmax(self.scale * p)  # (BS, N, C)
            ocr_context = tf.linalg.matmul(p, f, transpose_a=True)  # (BS, C, C2)

        else:
            BS, C, H, W = probabilities.shape
            p = tf.keras.layers.Reshape((C, -1))(probabilities)  # (BS, C, N)
            f = tf.keras.layers.Reshape((features.shape[1], -1))(features)  # (BS, C2, N)

            p = self.softmax(self.scale * p)  # (BS, C, N)
            ocr_context = tf.linalg.matmul(p, f, transpose_b=True)  # (BS, C, C2)

        return ocr_context


class ObjectAttentionBlock2D(tf.keras.layers.Layer):
    def __init__(self, filters, scale=1.0):
        super(ObjectAttentionBlock2D, self).__init__()
        self.filters = filters
        self.scale = scale

        self.max_pool2d = tf.keras.layers.MaxPooling2D(pool_size=(scale, scale))
        self.f_pixel = tf.keras.models.Sequential([
                                                    ConvolutionBnActivation(filters, (1, 1)),
                                                    ConvolutionBnActivation(filters, (1, 1))
        ])
        self.f_up = ConvolutionBnActivation(filters, (1, 1))

        self.softmax = tf.keras.layers.Activation("softmax")
        self.upsampling2d = tf.keras.layers.UpSampling2D(size=scale, interpolation="bilinear")

    def call(self, feats, ctx, training=None):
        if K.image_data_format() == "channels_last":
            # feats-dim: (BS, H, W, C) & ctx-dim: (BS, C, C2)
            ctx = tf.keras.layers.Permute((2, 1))(ctx)
            BS, H, W, C = feats.shape
            if self.scale > 1:
                feats = self.pool(feats, training=training)

            query = self.f_pixel(feats, training=training)              # (BS, H, W, C)
            query = tf.keras.layers.Reshape((-1, C))(query)             # (BS, N, C)
            # key = self.f_object(ctx, training=training)                 # (BS, C2, C)
            key = tf.keras.layers.Reshape((-1, C))(ctx)                 # (BS, C2, C)
            # value = self.f_down(ctx, training=training)                 # (BS, C2, C)
            value = tf.keras.layers.Reshape((-1, C))(ctx)               # (BS, C2, C)

            sim_map = tf.linalg.matmul(query, key, transpose_b=True)    # (BS, N, C2)
            sim_map = (self.filters ** -0.5) * sim_map                  # (BS, N, C2)
            sim_map = self.softmax(sim_map)                             # (BS, N, C2)

            context = tf.linalg.matmul(sim_map, value)                   # (BS, N, C)
            context = tf.keras.layers.Reshape((H, W, C))(context)       # (BS, H, W, C)
            context = self.f_up(context, training=training)             # (BS, H, W, C)
            if self.scale > 1:
                context = self.upsampling2d(context)

        else:
            # feats-dim: (BS, C, H, W) & ctx-dim: (BS, C, C2)
            BS, C, H, W = feats.shape
            if self.scale > 1:
                feats = self.pool(feats, training=training)

            query = self.f_pixel(feats, training=training)              # (BS, C, H, W)
            query = tf.keras.layers.Reshape((C, -1))(query)             # (BS, C, N)
            # key = self.f_object(ctx, training=training)                 # (BS, C, C2)
            key = tf.keras.layers.Reshape((C, -1))(ctx)                 # (BS, C, C2)
            # value = self.f_down(ctx, training=training)                 # (BS, C, C2)
            value = tf.keras.layers.Reshape((C, -1))(ctx)               # (BS, C, C2)

            sim_map = tf.linalg.matmul(query, key, transpose_a=True)    # (BS, N, C2)
            sim_map = (self.filters ** -0.5) * sim_map                  # (BS, N, C2)
            sim_map = self.softmax(sim_map)                             # (BS, N, C2)

            context = tf.linalg.matmul(sim_map, value, transpose_b=True) # (BS, N, C)
            context = tf.keras.layers.Permute(2, 1)(context)            # (BS, C, N)
            context = tf.keras.layers.Reshape((C, H, W))(context)       # (BS, C, H, W)
            context = self.f_up(context, training=training)             # (BS, C, H, W)
            if self.scale > 1:
                context = self.upsampling2d(context)

        return context


class SpatialOCRModule(tf.keras.layers.Layer):
    def __init__(self, filters, scale=1.0, dropout=0.1):
        super(SpatialOCRModule, self).__init__()
        self.filters = filters
        self.scale = scale
        self.dropout = dropout

        self.object_attention = ObjectAttentionBlock2D(filters, scale)

        axis = 3 if K.image_data_format() == "channels_last" else 1
        self.concat = tf.keras.layers.Concatenate(axis=axis)
        self.conv1x1_bn_relu = ConvolutionBnActivation(filters, (1, 1))
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, features, ocr_context, training=None):
        # features-dim: (BS, H, W, C) & ocr_context-dim: (BS, C, C2) (if K.image_data_format() == "channels_last")
        context = self.object_attention(features, ocr_context, training=training)  # (BS, H, W, C)

        output = self.concat([context, features])  # (BS, H, W, 2*C)
        output = self.conv1x1_bn_relu(output, training=training)  # (BS, H, W, C)
        output = self.dropout(output, training=training)  # (BS, H, W, C)

        return output
