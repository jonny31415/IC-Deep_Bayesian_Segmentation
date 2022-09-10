import tensorflow as tf

def get_resnet(input_shape, trunk_name, output_stride=None):

    if trunk_name == 'ResNet50':
        trunk = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape)
    else:
        raise NotImplementedError

    # if output_stride == 8:
    #     for n, m in self.layer3.named_modules():
    #         if 'conv2' in n:
    #             m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
    #         elif 'downsample.0' in n:
    #             m.stride = (1, 1)
    #     for n, m in self.layer4.named_modules():
    #         if 'conv2' in n:
    #             m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
    #         elif 'downsample.0' in n:
    #             m.stride = (1, 1)

    if output_stride == 8:
        input('entrou')
        for i, layer in enumerate(trunk.layers):
            for node in layer._outbound_nodes:
                layer_name = node.outbound_layer.name
                if layer_name.endswith('2_conv'):
                    if layer_name.startswith('conv4'):
                        trunk.get_layer(layer_name).dilation_rate = (2, 2)
                        trunk.get_layer(layer_name).strides = (1, 1)
                        trunk.layers[i] = tf.keras.Sequential([tf.keras.layers.ZeroPadding2D(padding=(2, 2)),
                                                               trunk.layers[i]])
                    elif layer_name.startswith('conv5'):
                        trunk.get_layer(layer_name).dilation_rate = (4, 4)
                        trunk.get_layer(layer_name).strides = (1, 1)
                        trunk.layers[i] = tf.keras.Sequential([tf.keras.layers.ZeroPadding2D(padding=(4, 4)),
                                                               trunk.layers[i]])

    return trunk