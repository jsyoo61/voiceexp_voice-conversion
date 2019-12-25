import tensorflow as tf

def gated_linear_layer(inputs, gates, name = None):

    activation = tf.multiply(x = inputs, y = tf.sigmoid(gates), name = name)

    return activation

def instance_norm_layer(
    inputs,
    epsilon = 1e-06,
    activation_fn = None,
    name = None):

    instance_norm_layer = tf.contrib.layers.instance_norm(
        inputs = inputs,
        epsilon = epsilon,
        activation_fn = activation_fn)

    return instance_norm_layer

def conv1d_layer(
    inputs,
    filters,
    kernel_size,
    strides = 1,
    padding = 'same',
    activation = None,
    kernel_initializer = None,
    name = None):

    conv_layer = tf.layers.conv1d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        activation = activation,
        kernel_initializer = kernel_initializer,
        name = name)

    return conv_layer

def conv2d_layer(
    inputs,
    filters,
    kernel_size,
    strides,
    padding = 'same',
    activation = None,
    kernel_initializer = None,
    name = None):

    conv_layer = tf.layers.conv2d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        activation = activation,
        kernel_initializer = kernel_initializer,
        name = name)

    return conv_layer

def residual1d_block(
    inputs,
    filters = 1024,
    kernel_size = 3,
    strides = 1,
    name_prefix = 'residule_block_'):

    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')
    h2 = conv1d_layer(inputs = h1_glu, filters = filters // 2, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h2_conv')
    h2_norm = instance_norm_layer(inputs = h2, activation_fn = None, name = name_prefix + 'h2_norm')

    h3 = inputs + h2_norm

    return h3

def downsample1d_block(
    inputs,
    filters,
    kernel_size,
    strides,
    name_prefix = 'downsample1d_block_'):

    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def downsample2d_block(
    inputs,
    filters,
    kernel_size,
    strides,
    name_prefix = 'downsample2d_block_'):

    h1 = conv2d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv2d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def upsample1d_block(
    inputs,
    filters,
    kernel_size,
    strides,
    shuffle_size = 2,
    name_prefix = 'upsample1d_block_'):

    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_shuffle = pixel_shuffler(inputs = h1, shuffle_size = shuffle_size, name = name_prefix + 'h1_shuffle')
    h1_norm = instance_norm_layer(inputs = h1_shuffle, activation_fn = None, name = name_prefix + 'h1_norm')

    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_shuffle_gates = pixel_shuffler(inputs = h1_gates, shuffle_size = shuffle_size, name = name_prefix + 'h1_shuffle_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_shuffle_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')

    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def dense_layer(
    inputs,
    units,
    activation = None,
    kernel_initializer = None,
    name = None
    ):

    dense_layer = tf.layers.dense(
        inputs = inputs,
        units = units,
        activation = activation,
        kernel_initializer = kernel_initializer,
        name = name)

    return dense_layer


def pixel_shuffler(inputs, shuffle_size = 2, name = None):

    n = tf.shape(inputs)[0]
    w = tf.shape(inputs)[1]
    c = inputs.get_shape().as_list()[2]

    oc = c // shuffle_size
    ow = w * shuffle_size

    outputs = tf.reshape(tensor = inputs, shape = [n, ow, oc], name = name)

    return outputs

def generator(inputs, reuse = False, scope_name = 'generator'):

    # inputs has shape [batch_size, num_features, time]
    # we need to convert it to [batch_size, time, num_features]
    inputs = tf.transpose(inputs, perm = [0, 2, 1], name = 'input_transpose')

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        # Hidden layers
        h1 = dense_layer(inputs = inputs, units = 128, activation = tf.sigmoid, name = 'h1_dense')
        h2 = dense_layer(inputs = h1, units = 256, activation = tf.sigmoid, name = 'h2_dense')
        h3 = dense_layer(inputs = h2, units = 256, activation = tf.sigmoid, name = 'h3_dense')
        h4 = dense_layer(inputs = h3, units = 128, activation = tf.sigmoid, name = 'h4_dense')

        # Output
        o1 = dense_layer(inputs = h4, units = inputs.shape[2], activation = None, name = 'output')
        o2 = tf.transpose(o1, perm = [0, 2, 1], name = 'output_transpose')

    return o2


def discriminator(inputs, reuse = False, scope_name = 'discriminator'):

    # inputs has shape [batch_size, num_features, time]
    # we need to convert it to [batch_size, time, num_features]
    inputs = tf.transpose(inputs, perm = [0, 2, 1], name = 'input_transpose')

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        # Hidden layers
        h1 = dense_layer(inputs = inputs, units = 128, activation = tf.sigmoid, name = 'h1_dense')
        h2 = dense_layer(inputs = h1, units = 256, activation = tf.sigmoid, name = 'h2_dense')
        h3 = dense_layer(inputs = h2, units = 256, activation = tf.sigmoid, name = 'h3_dense')
        h4 = dense_layer(inputs = h3, units = 128, activation = tf.sigmoid, name = 'h4_dense')

        # Output
        o1 = dense_layer(inputs = h4, units = 1, activation = tf.nn.sigmoid, name = 'output')

        return o1
