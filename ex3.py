import numpy as np
from random import shuffle
from abc import ABC, abstractmethod


class IModel(ABC):
    @abstractmethod
    def predict(self, row):
        raise NotImplementedError

    @abstractmethod
    def train(self, row, row_y_values, learn_rate):
        raise NotImplementedError

# TODO: -- call me when you start :) --
# TODO: -- Borreda -- turn this into a class: activationFuncDerivatives
activationFuncDerivativs = {"sigmoid": lambda x: x*(1-x),
                      "tanH": lambda x: 4/np.power((np.exp(-x) + np.exp(x)), 2),
                      "reLU": reLU_Deriv,
                      "PreLU" : PreLU_deriv}


class ActivationFunctions:
    _functions_map = dict()

    def __init__(self):
        self._functions_map["sigmoid"] = self.sigmoid_func
        self._functions_map["tan_h"] = self.tan_h_func
        self._functions_map["rel_u"] = self.rel_u
        self._functions_map["pre_lu"] = self.pre_lu

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
            if x[i] < 0: x[i] = 0

        return x

    @staticmethod
    def pre_lu(x):
        np.maximum(np.multiply(0.01, x), x)


class FitModel:
    def __init__(self, train_x_data, train_y_data):
        self._X = train_x_data
        self._Y = train_y_data

    def fit_model(self, model, epochs, learning_rate, print_results=False):
        """
        train(model, train_x_data, train_y_values, epochs, learning_rate, verbose=False).

        :param model: a given ModelInterface
        :param epochs: given epochs list
        :param learning_rate: learning rate for the networks
        :param print_results: true in order to print the results, or false otherwise. False default.

        :return: the best model found
        """
        # TODO: -- Alon -- ("train")
        # TODO: -- Alon -- also "forward" function
        pass

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

    # TODO: -- borreda -- add these two to the FitModel class
    def backProp(input, softMax_deriv, hiddenFunc_deriv, forwardVals, params):
        weights2_grad = np.dot(softMax_deriv, forwardVals["h1"].T)
        bias2_grad = softMax_deriv
        bias1_temp1 = np.dot(params["weights2"].T, softMax_deriv)
        bias1_grad = bias1_temp1 * hiddenFunc_deriv(forwardVals["h1"])
        weights1_grad = np.dot(bias1_grad, input[0].T)
        return {"w1_grad": weights1_grad, "b1_grad": bias1_grad, "w2_grad": weights2_grad, "b2_grad": bias2_grad}

    # updating the params
    def update_params(params, LR, gradients):
        params["weights1"] = params["weights1"] - LR["weights1"] * gradients["w1_grad"]
        params["bias1"] = params["bias1"] - LR["bias1"] * gradients["b1_grad"]
        params["weights2"] = params["weights2"] - LR["weights2"] * gradients["w2_grad"]
        params["bias2"] = params["bias2"] - LR["bias2"] * gradients["b2_grad"]
        return params


class DataManipulations:
    @staticmethod
    def load_text(file_path):
        try:
            return np.loadtxt(file_path)
        except IOError:
            print("The file path provided is corrupted, or file does not exist.")
            return list()


TRAIN_X_FP = "train_x"
TRAIN_Y_FP = "train_y"

VALIDATION_PORTION = 1/5


def load_training_data():
    """
    load_training_data().

    :return: train_x and train_y after reading the files
    """
    return DataManipulations.load_text(TRAIN_X_FP), DataManipulations.load_text(TRAIN_Y_FP)


def zip_data(segment_a, segment_b):
    """
    zip_data(segment_a, segment_b).

    :param segment_a: segment a of the data to zip
    :param segment_b: segment b of the data to zip
    :return: zip of the segments
    """
    return zip(segment_a, segment_b)


def main():
    # TODO: -- Alon -- add the main...
    pass


if __name__ == '__main__':
    train_x, train_y = load_training_data()

    # zip files
    training_data = zip_data(train_x, train_y)

    # shuffle training data
    shuffle(training_data)

    # split to validation and actual_train_size
    train_size = len(train_x)
    validation_size = train_size * VALIDATION_PORTION
    actual_train_size = train_size * (1 - VALIDATION_PORTION)

    actual_train_inputs = training_data[:actual_train_size:]
    validation_inputs = training_data[actual_train_size::]

    # TODO: -- Alon -- add test_forward call to here (code above main)