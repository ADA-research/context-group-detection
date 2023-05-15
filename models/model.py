import keras as keras
import tensorflow as tf
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Concatenate, Lambda, BatchNormalization, Flatten
from keras.models import Model


def conv(filters, reg, name=None):
    return Conv2D(filters=filters, kernel_size=1, padding='valid', kernel_initializer="he_normal",
                  use_bias='True', kernel_regularizer=reg, activation=tf.nn.relu, name=name)


def build_model(reg_amt, drop_amt, context_size, features, global_filters,
                individual_filters, combined_filters, consecutive_frames, no_pointnet=False, symmetric=False):
    context_inputs = keras.layers.Input(shape=(context_size, consecutive_frames, features), name='context')
    pair_inputs = keras.layers.Input(shape=(2, consecutive_frames, features), name='pair')

    reg = keras.regularizers.l2(reg_amt)

    # TODO check how to use LSTM
    #  one for each agent
    #  one for pair and one for context
    pair_lstm = keras.layers.LSTM()
    context_lstm = keras.layers.LSTM()

    # Dyad Transform
    for filters in individual_filters:
        pair_conv_out = conv(filters, reg)(pair_inputs)
        pair_drop_out = Dropout(drop_amt)(pair_conv_out)
        pair_batch_norm_out = BatchNormalization()(pair_drop_out)

    y_0 = Lambda(lambda input: tf.slice(input, [0, 0, 0, 0], [-1, -1, 1, -1]))(pair_batch_norm_out)
    y_1 = Lambda(lambda input: tf.slice(input, [0, 0, 1, 0], [-1, -1, 1, -1]))(pair_batch_norm_out)

    if no_pointnet:
        combined = Concatenate(name='concat')([Flatten()(y_0), Flatten()(y_1)])
    else:

        # Context Transform
        for filters in global_filters:
            context_conv_out = conv(filters, reg)(context_inputs)
            context_drop_out = Dropout(drop_amt)(context_conv_out)
            context_batch_norm_out = BatchNormalization()(context_drop_out)

        context_max_pool_out = MaxPooling2D(name="global_pool", pool_size=[1, context_size], strides=1,
                                            padding='valid')(context_batch_norm_out)
        context_drop_out = Dropout(drop_amt)(context_max_pool_out)
        context_batch_norm_out = BatchNormalization()(context_drop_out)
        context_flat = Flatten()(context_batch_norm_out)

        # enforce symmetric affinity predictions by doing pointnet on 2 people
        if symmetric:
            pair_max_pool_out = MaxPooling2D(name="symmetric_pool", pool_size=[1, 2], strides=1, padding='valid')(
                pair_batch_norm_out)
            pair_flat = Flatten()(pair_max_pool_out)
            combined = Concatenate(name='concat')([context_flat, pair_flat])
        else:
            combined = Concatenate(name='concat')([context_flat, Flatten()(y_0), Flatten()(y_1)])

    # Final MLP from paper
    for filters in combined_filters:
        combined_dense_out = Dense(units=filters, use_bias='True', kernel_regularizer=reg, activation=tf.nn.relu,
                                   kernel_initializer="he_normal")(combined)
        combined_drop_out = Dropout(drop_amt)(combined_dense_out)
        combined_batch_norm = BatchNormalization()(combined_drop_out)

    # final pred
    affinity = Dense(units=1, use_bias="True", kernel_regularizer=reg, activation=tf.nn.sigmoid,
                     name='affinity', kernel_initializer="glorot_normal")(combined_batch_norm)

    model = Model(inputs=[context_inputs, pair_inputs], outputs=affinity)

    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=1e-5, amsgrad=False, clipvalue=0.5)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['mse'])

    return model
