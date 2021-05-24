import tensorflow as tf
from keras.layers import LSTM, Dense, Flatten
from keras.layers import Input, Reshape, Permute, concatenate
from keras.layers import Layer, Dropout, Masking
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

class ReNetLayer(Layer):
    def __init__(self, size_of_patches, hidden_size,use_dropout=False, dropout_rate=None, is_first_layer=False):

        super().__init__()

        self.size_of_patches = size_of_patches
        self.w_p = size_of_patches[0][0]
        self.h_p = size_of_patches[0][1]

        self.is_first_layer = is_first_layer
        if is_first_layer:
            self.mask = Masking(mask_value=-100.0)

        self.hidden_size = hidden_size
        self.LSTM_up_down = LSTM(self.hidden_size, return_sequences=True)
        self.LSTM_down_up = LSTM(self.hidden_size, return_sequences=True, go_backwards=True)
        self.LSTM_left_right = LSTM(self.hidden_size, return_sequences=True)
        self.LSTM_right_left = LSTM(self.hidden_size, return_sequences=True, go_backwards=True)

        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout_up_down = Dropout(dropout_rate)
            self.dropout_down_up = Dropout(dropout_rate)
            self.dropout_left_right = Dropout(dropout_rate)
            self.dropout_right_left = Dropout(dropout_rate)

        self.layer_horizontal_activations_permutarion = Permute((2, 1, 3))

        self.lstm_weights_array_len = 3

    def build(self, input_shape): # Create a trainable weight variable for this layer.
        super().build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        J = int(input_shape[1]) // self.h_p
        I = int(input_shape[2]) // self.w_p

        return (input_shape[0], J, I, 2 * self.hidden_size)

    def get_weights(self):
        weights = []
        weights.extend(self.LSTM_up_down.get_weights())
        weights.extend(self.LSTM_down_up.get_weights())
        weights.extend(self.LSTM_left_right.get_weights())
        weights.extend(self.LSTM_right_left.get_weights())
        self.lstm_weights_array_len = len(self.LSTM_up_down.get_weights())

        return weights

    def set_weights(self, weights):
        if len(weights) != 4 * self.lstm_weights_array_len:
            raise Exception("wrong number of weights matrixes")

        self.lstm_weights_array_len
        self.LSTM_up_down.set_weights(weights[:self.lstm_weights_array_len])
        self.LSTM_down_up.set_weights(weights[self.lstm_weights_array_len:2 * self.lstm_weights_array_len])
        self.LSTM_left_right.set_weights(weights[2 * self.lstm_weights_array_len:3 * self.lstm_weights_array_len])
        self.LSTM_right_left.set_weights(weights[3 * self.lstm_weights_array_len:])

    def call(self, inputs):
        self.__validate_patch_size(inputs.shape)
        self.__initialize_input_size_dependent_variables(inputs.shape)

        vertical_sweep_output = self.vertical_sweep(inputs)
        horizontal_sweep_output = self.horizontal_sweep(vertical_sweep_output)

        return horizontal_sweep_output

    def __validate_patch_size(self, inputs_shape):
        if inputs_shape[2] % self.h_p != 0 or inputs_shape[1] % self.w_p != 0:
            raise ValueError(
            "invalid patches size ({}) "
            "for current image size ({}). "
            "Please resizie image ".format(self.size_of_patches, inputs_shape[1:]))

    def __initialize_input_size_dependent_variables(self, inputs_shape):
        self.J = int(inputs_shape[1]) // self.h_p
        self.I = int(inputs_shape[2]) // self.w_p

        self.layer_vertical_patches_reshape = Reshape((self.J, self.w_p * self.h_p * int(inputs_shape[3])))
        self.layer_precise_tensor_shape = Reshape((self.J, self.I, int(2 * self.hidden_size)))

    def vertical_sweep(self, inputs):
        LSTM_outputs = []

        for col in self.get_columns(inputs):
            column_activations = self.get_activations_for_column(col)
            LSTM_outputs.append(column_activations)

        merged = concatenate(LSTM_outputs, axis=2)
        result = self.layer_precise_tensor_shape(merged)
        return result

    def get_columns(self, inputs):
        for j in range(0, inputs.shape[2], self.w_p):
            yield inputs[:, :, j:j + self.w_p, :]

    def get_activations_for_column(self, col):
        patches = self.convert_to_vertical_patch(col)
        p = self.mask_if_first_layer(patches)
        up_down_activation, down_up_activation = self.calc_vertical_LSTM_activations(p)
        return self.merge_vert_LSTM_activations(up_down_activation, down_up_activation)

    def convert_to_vertical_patch(self, column):
        return self.layer_vertical_patches_reshape(column)

    def mask_if_first_layer(self, patches):
        if self.is_first_layer:
            return self.mask(patches)
        else:
            return patches

    def calc_vertical_LSTM_activations(self, patches):
        up_down_activation = self.LSTM_up_down(patches)
        down_up_activation = self.LSTM_down_up(patches)

        if self.use_dropout:
            up_down_activation = self.dropout_up_down(up_down_activation)
            down_up_activation = self.dropout_down_up(down_up_activation)

        return up_down_activation, down_up_activation

    def merge_vert_LSTM_activations(self, first_tensor, second_tensor):
        merged_vector = concatenate([tf.keras.backend.expand_dims(first_tensor, axis=2),
                                     tf.keras.backend.expand_dims(second_tensor, axis=2)], axis = 3)
        return merged_vector

    def horizontal_sweep(self, inputs):
        LSTM_outputs = []

        for row in self.get_rows(inputs):
            row_activations = self.get_activations_for_row(row)
            LSTM_outputs.append(row_activations)

        merged = concatenate(LSTM_outputs, axis=1)
        return self.layer_precise_tensor_shape(merged)

    def get_rows(self, inputs):
        for i in range(0, inputs.shape[1]):
            yield inputs[:, i:i + 1, :, :]

    def get_activations_for_row(self, row):
        patches = self.get_hor_patches(row)

        left_right_activations, right_left_activations = self.calc_horizontal_LSTM_activations(patches)

        merged_tensor = self.merge_hor_LSTM_activations(left_right_activations, right_left_activations)
        return self.layer_horizontal_activations_permutarion(merged_tensor)

    def get_hor_patches(self, row):
        return tf.squeeze(row, axis=1)

    def calc_horizontal_LSTM_activations(self, patches):
        left_right_activations = self.LSTM_left_right(patches)
        right_left_activations = self.LSTM_right_left(patches)

        if self.use_dropout:
            left_right_activations = self.dropout_left_right(left_right_activations)
            right_left_activations = self.dropout_right_left(right_left_activations)
        return left_right_activations, right_left_activations

    def merge_hor_LSTM_activations(self, first_tensor, second_tensor):
        merged_vector = concatenate([tf.keras.backend.expand_dims(first_tensor, axis=2),
                                     tf.keras.backend.expand_dims(second_tensor, axis=2)], axis = 3)
        return merged_vector
        
        
     $#################################################################
     
     
from keras.models import Sequential
from ReNet.ReNetLayer import *


def get_cifar10_model():
    model = Sequential()

    reNet_hidden_size = 320
    model.add(ReNetLayer([[2, 2]], reNet_hidden_size, use_dropout=True, dropout_rate=0.1, is_first_layer=True))
    model.add(ReNetLayer([[2, 2]], reNet_hidden_size,  use_dropout=True, dropout_rate=0.1))
    model.add(ReNetLayer([[2, 2]], reNet_hidden_size, use_dropout=True, dropout_rate=0.1))

    model.add(Flatten())
    fully_conn_hidden_size = 4096
    model.add(Dense(fully_conn_hidden_size, activation='relu'))
    model.add(Dropout(0.1))

    num_classes = 10
    model.add(Dense(num_classes, activation='softmax'))

    return model
