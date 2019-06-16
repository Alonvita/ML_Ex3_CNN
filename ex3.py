import numpy as np
from random import shuffle
from abc import ABC, abstractmethod


class IModel(ABC):
    @abstractmethod
    def predict(self, row):
        raise NotImplementedError

    @abstractmethod
    def update_model(self, gradients, learning_rate):
        raise NotImplementedError


class Cnn(IModel):
    def __init__(self, hidden_layer_size,
                 input_size, output_size,
                 activation_function, activation_derivative):
        # initialize currently used function and derivative to None
        self._strategy_func = activation_function
        self._derivative_strategy_func = activation_derivative

        self._model_structure = {
            "Layer1": np.random.uniform(-0.08, 0.08, [hidden_layer_size, input_size]),
            "Bias1": np.random.uniform(-0.08, 0.08, [hidden_layer_size, 1]),
            "Layer2": np.random.uniform(-0.08, 0.08, [output_size, hidden_layer_size]),
            "Bias2": np.random.uniform(-0.08, 0.08, [output_size, 1])
        }

    def set_strategy_func(self, func):
        self._strategy_func = func

    def set_derivative_strategy_func(self, func):
        self._derivative_strategy_func = func

    def predict(self, row):
        """
        predict(self, row).

        :param row: data row.

        :return: a dict containing the desired forward values
        """

        z1 = np.dot(self._model_structure["Layer1"], row) + self._model_structure["Bias1"]

        # apply function on first level
        h1 = self._strategy_func(z1)

        # calculate output vector
        output_vector = np.dot(self._model_structure["Layer2"], h1) + self._model_structure["Bias2"]

        probabilities_vector = ActivationFunctions.soft_max(output_vector)

        # return y_hat
        return probabilities_vector, h1

    def back_propagation(self, row, soft_max_derivative, last_prediction_values):
        h1 = last_prediction_values[1]

        # last prediction values [1] is h1
        layer2_grad = np.dot(soft_max_derivative, h1.T)
        bias2_grad = soft_max_derivative

        bias1_temp1 = np.dot(self._model_structure["Layer2"].T, soft_max_derivative)
        bias1_grad = bias1_temp1 * self._derivative_strategy_func(h1)

        layer1_grad = np.dot(bias1_grad, row.T)

        return {
            "w1_grad": layer1_grad,
            "b1_grad": bias1_grad,
            "w2_grad": layer2_grad,
            "b2_grad": bias2_grad
        }

    def update_model(self, gradients, learning_rate):
        self._model_structure["Layer1"] = \
            self._model_structure["Layer1"] - learning_rate * gradients["w1_grad"]
        self._model_structure["Bias1"] = \
            self._model_structure["Bias1"] - learning_rate * gradients["b1_grad"]
        self._model_structure["Layer2"] = \
            self._model_structure["Layer2"] - learning_rate * gradients["w2_grad"]
        self._model_structure["Bias2"] = \
            self._model_structure["Bias2"] - learning_rate * gradients["b2_grad"]

        # return the model structure AFTER the update
        return self._model_structure


class ActivationFunctions:
    _functions_map = dict()
    _derivatives_map = dict()

    @staticmethod
    def sigmoid_func(x):
        exp_x = np.exp(-x)

        return 1 / (1 + exp_x)

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def soft_max(x):
        """
        soft_max(x).

        :param x: n-dim numpy array.
        :return: softmax values as n-dim numpy array
        """
        shifted_x = np.subtract(x, np.amax(x))
        exp_values = np.exp(shifted_x)

        return np.divide(exp_values, np.sum(exp_values))


class FitModel:
    def __init__(self, train_x_data, train_y_data, validation_portion):
        # zip files and shuffle
        zipped_files = list(DataManipulations.zip_data(train_x_data, train_y_data))

        # TODO: add this again -- shuffle(zipped_files)

        self._shuffled_pre_split_data = zipped_files

        # split to actual and validation
        self.__split_to_actual_train_and_validation_data(len(train_x_data), validation_portion)

    def __split_to_actual_train_and_validation_data(self, data_length, validation_portion):
        actual_train_size = int(data_length * (1 - validation_portion))

        self._actual_train_data = self._shuffled_pre_split_data[:actual_train_size]
        self._validation_data = self._shuffled_pre_split_data[actual_train_size:]

    def get_actual_data_created(self):
        return self._actual_train_data

    def get_validation_data_created(self):
        return self._validation_data

    def fit_model(self, model, epochs, learning_rate, debugging=False):
        """
        train(model, train_x_data, train_y_values, epochs, learning_rate, verbose=False).

        :param model: a given ModelInterface
        :param epochs: given epochs list
        :param learning_rate: learning rate for the networks
        :param debugging: true in order to print the results, or false otherwise. False default.
        """
        for epochs in range(epochs):
            total_loss = 0.0

            # shuffle actual data
            # TODO: add this again shuffle(self._actual_train_data)

            model_after_train = None

            for row, y_val in self._actual_train_data:
                prediction_dict = model.predict(row.reshape(784, 1))

                # predict for the current, reshaped, row
                y_hat = prediction_dict[0]

                # sum the loss
                total_loss -= np.log(y_hat[int(y_val)])

                # calculate y_hat
                y_hat[int(y_val)] -= 1

                # back propagation
                gradients = model.back_propagation(row.reshape(784, 1), y_hat, prediction_dict)

                # update the model
                model_after_train = model.update_model(gradients, learning_rate)

            if debugging and model_after_train is not None:
                validation_loss, acc = \
                    self.__compare_to_validation(model)

                print(
                    "Epoch: {}\nAVG train loss: {}\nAVG validation loss: {}\n Accuracy percent: {}\n".format(
                        epochs, total_loss / len(self._actual_train_data), validation_loss, acc * 100))

    def __compare_to_validation(self, model):
        sum_loss = 0.0
        hits = 0.0
        size = len(self._validation_data)

        for row, y_val in self._actual_train_data:

            prediction_tup = model.predict(row.reshape(784, 1))

            y_hat = prediction_tup[0]

            if y_hat.argmax() == int(y_val):
                hits += 1

        acc = hits / size
        avg_loss = sum_loss / size

        return avg_loss, acc


class DataManipulations:
    @staticmethod
    def load_text(file_path):
        try:
            return np.loadtxt(file_path)
        except IOError:
            print("The file path provided is corrupted, or file does not exist.")
            return list()

    @staticmethod
    def zip_data(segment_a, segment_b):
        """
        zip_data(segment_a, segment_b).

        :param segment_a: segment a of the data to zip
        :param segment_b: segment b of the data to zip
        :return: zip of the segments
        """
        return zip(segment_a, segment_b)


TRAIN_X_FP = "xx"
TRAIN_Y_FP = "yy"


def load_training_data():
    """
    load_training_data().

    :return: train_x and train_y after reading the files
    """
    return DataManipulations.load_text(TRAIN_X_FP), DataManipulations.load_text(TRAIN_Y_FP)


OUTPUT_SIZE = 10
INPUT_SIZE = 784
HIDDEN_LAYER_SIZE = 100
VALIDATION_PORTION = 1/5
EPOCHS = 30

LEARNING_RATE = 0.01


if __name__ == '__main__':
    train_x, train_y = load_training_data()

    cnn_model = Cnn(HIDDEN_LAYER_SIZE,
                    INPUT_SIZE, OUTPUT_SIZE,
                    ActivationFunctions.sigmoid_func,
                    ActivationFunctions.sigmoid_derivative)

    # create the FitModel
    model_trainer = FitModel(train_x, train_y, VALIDATION_PORTION)

    # go.
    model_trainer.fit_model(cnn_model, EPOCHS, LEARNING_RATE, debugging=True)
