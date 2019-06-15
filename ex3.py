import numpy as np
from random import shuffle
from abc import ABC, abstractmethod


class IModel(ABC):
    @abstractmethod
    def predict(self, row):
        raise NotImplementedError

    @abstractmethod
    def update_model(self, gradients, learn_rate):
        raise NotImplementedError


class Cnn(IModel):
    def __init__(self, hidden_layer_size,
                 input_size, output_size,
                 activation_function, activation_derivative):
        # initialize currently used function and derivative to None
        self._strategy_func = None
        self._derivative_strategy_func = None

        self._model_structure = {
            "Layer1": np.random.uniform(-0.08, 0.08, [hidden_layer_size, input_size]),
            "Bias1": np.random.uniform(-0.08, 0.08, [hidden_layer_size, 1]),
            "Layer2": np.random.uniform(-0.08, 0.08, [output_size, hidden_layer_size]),
            "Bias2": np.random.uniform(-0.08, 0.08, [output_size, 1])
        }

        # initialized strategy func and derivative func.
        self.set_strategy_func(activation_function)
        self.set_derivative_strategy_func(activation_derivative)

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

        # apply function on first level
        h1 = self._strategy_func(
            np.dot(self._model_structure["Layer1"], row) +
            self._model_structure["bias1"])

        # calculate output vector
        output_vector = \
            self._strategy_func(
                np.dot(self._model_structure["Layer2"], h1) +
                self._model_structure["bias2"])

        probabilities_vector = ActivationFunctions.soft_max(output_vector)

        # return y_hat
        return {
            probabilities_vector: "prob_vec",
            h1: "h1"
        }

    def back_propagation(self, row, soft_max_derivative, last_prediction_values):
        layer2_grad = np.dot(soft_max_derivative, last_prediction_values["h1"].T)
        bias2_grad = soft_max_derivative

        bias1_temp1 = np.dot(self._model_structure["Layer2"].T, soft_max_derivative)
        bias1_grad = bias1_temp1 * self._derivative_strategy_func(last_prediction_values["h1"])

        layer1_grad = np.dot(bias1_grad, row.T)

        return {
            "w1_grad": layer1_grad,
            "b1_grad": bias1_grad,
            "w2_grad": layer2_grad,
            "b2_grad": bias2_grad
        }

    def update_model(self, gradients, learn_rate):
        self._model_structure["Layer1"] = \
            self._model_structure["Layer1"] - learn_rate["Layer1"] * gradients["w1_grad"]
        self._model_structure["bias1"] = \
            self._model_structure["bias1"] - learn_rate["bias1"] * gradients["b1_grad"]
        self._model_structure["weights2"] = \
            self._model_structure["weights2"] - learn_rate["weights2"] * gradients["w2_grad"]
        self._model_structure["bias2"] = \
            self._model_structure["bias2"] - learn_rate["bias2"] * gradients["b2_grad"]

        # return the model structure AFTER the update
        return self._model_structure


class ActivationFunctions:
    _functions_map = dict()
    _derivatives_map = dict()

    @staticmethod
    def sigmoid_func(x):
        return x / (1 + np.exp(x))

    @staticmethod
    def tan_h_func(x):
        return np.tanh(x)

    @staticmethod
    def rel_u(x):
        length = len(x)

        for i in range(length):
            if x[i] < 0:
                x[i] = 0

        return x

    @staticmethod
    def pre_lu(x):
        np.maximum(np.multiply(0.01, x), x)

    @staticmethod
    def soft_max(vector):
        shifted_x = np.exp(vector - vector.max())
        exp_values = np.exp(shifted_x)

        return np.divide(exp_values, np.sum(exp_values))


class FitModel:
    def __init__(self, train_x_data, train_y_data, validation_portion):
        # zip files and shuffle
        self._shuffled_training_data = shuffle(DataManipulations.zip_data(train_x_data, train_y_data))

        # split to actual and validation
        self.__split_to_actual_train_and_validation_data(len(train_x_data), validation_portion)

    def __split_to_actual_train_and_validation_data(self, data_length, validation_portion):
        actual_train_size = int(data_length * (1 - validation_portion))

        self._actual_train_data = self._shuffled_training_data[:actual_train_size]
        self._validation_data = self._shuffled_training_data[actual_train_size:]

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
            shuffle(self._actual_train_data)

            model_after_train = None

            train_size = len(self._actual_train_data[0])

            for index in range(train_size):
                row = self._actual_train_data[0][index]
                y_val = self._actual_train_data[1][index]

                prediction_dict = model.predict(row.reshape(784, 1))

                # predict for the current, reshaped, row
                y_hat = prediction_dict["prob_vec"]

                # sum the loss
                total_loss += - np.log(y_hat[y_val])

                # calculate y_hat
                y_hat[y_val] -= 1

                # back propagation
                gradients = model.back_propagation(row, y_hat, prediction_dict)

                # update the model
                model_after_train = model.update_model(learning_rate, gradients)

            if debugging and model_after_train is not None:
                validation_loss, acc = \
                    self.__compare_to_validation(model)

                print(
                    "Epoch: {}\nAVG train loss: {}\nAVG validation loss: {}\n Accuracy percent: {}\n".format(
                        epochs, total_loss / train_size, validation_loss, acc * 100))

    def meassure_accuracy(self):
        # TODO: -- borreda --
        # TODO: you can use our code from ex2...
        sum_loss = 0.0
        matches = 0.0
        size = len(validationInputs)
        for x in validationInputs:
            forwardVals = forward(x, params, hiddenFunc)
            sum_loss += forwardVals["loss"]
            if forwardVals["y_hat"].argmax() == x[1]:
                matches += 1
        acc = matches / size
        avg_loss = sum_loss / size
        return avg_loss, acc

    def __compare_to_validation(self, model):
        sum_loss = 0.0
        hits = 0.0
        size = len(self._validation_data)

        for index in range(len(self._validation_data)):
            row = self._validation_data[0]
            y_val = self._validation_data[1]

            prediction_dict = model.predict(row.reshape(784, 1))

            y_hat = prediction_dict["prob_vec"]

            if y_hat.argmax() == y_val:
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

    def zip_data(segment_a, segment_b):
        """
        zip_data(segment_a, segment_b).

        :param segment_a: segment a of the data to zip
        :param segment_b: segment b of the data to zip
        :return: zip of the segments
        """
        return zip(segment_a, segment_b)


TRAIN_X_FP = "train_x"
TRAIN_Y_FP = "train_y"


def load_training_data():
    """
    load_training_data().

    :return: train_x and train_y after reading the files
    """
    return DataManipulations.load_text(TRAIN_X_FP), DataManipulations.load_text(TRAIN_Y_FP)


OUTPUT_SIZE = 10
INPUT_SIZE = pow(28, 2)
HIDDEN_LAYER_SIZE = 100
VALIDATION_PORTION = 1/5


if __name__ == '__main__':
    train_x, train_y = load_training_data()

    cnn_model = Cnn(HIDDEN_LAYER_SIZE,
                    len(train_x), OUTPUT_SIZE,
                    ActivationFunctions.sigmoid_func,
                    ActivationFunctions.sigmoid_derivative)

    # create the FitModel
    fit_model = FitModel(train_x, train_y, VALIDATION_PORTION)

