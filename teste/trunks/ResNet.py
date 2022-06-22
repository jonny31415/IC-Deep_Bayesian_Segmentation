import tensorflow as tf

def get_resnet(input_shape, trunk_name, output_stride=None):

    if trunk_name == 'ResNet50':
        trunk = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape)
    else:
        raise NotImplementedError

    if output_stride == 8:
        for layer in trunk.layers:
            for node in layer._outbound_nodes:
                layer_name = node.outbound_layer.name
                if layer_name.endswith('2_conv'):
                    if layer_name.startswith('conv4'):
                        # TODO: add padding --> model.add(tf.keras.layers.ZeroPadding2D(padding=(2, 2)))
                        trunk.get_layer(layer_name).dilation_rate = (2, 2)
                        trunk.get_layer(layer_name).strides = (1, 1)
                    elif layer_name.startswith('conv5'):
                        # TODO: add padding --> model.add(tf.keras.layers.ZeroPadding2D(padding=(2, 2)))
                        trunk.get_layer(layer_name).dilation_rate = (4, 4)
                        trunk.get_layer(layer_name).strides = (1, 1)

    return trunk