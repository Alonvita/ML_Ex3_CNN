
import numpy as np
from random import shuffle

# softMax func
softMax = lambda vec: np.exp(vec - vec.max()) / np.exp(vec - vec.max()).sum()

# params initialization
input_size = 28*28  # size of images
hidden_size = 100   # hyper param, but needed it here for initialization
output_size = 10    # size of classifications
weights1 = np.random.uniform(-0.08, 0.08, [hidden_size, input_size])
bias1 = np.random.uniform(-0.08, 0.08, [hidden_size, 1])
weights2 = np.random.uniform(-0.08, 0.08, [output_size, hidden_size])
bias2 = np.random.uniform(-0.08, 0.08, [output_size, 1])
params = {"weights1": weights1, "bias1": bias1, "weights2": weights2, "bias2": bias2}

# reLU func and its deriv
def reLU_Deriv(x):
    zeros = np.array([0 for i in range(hidden_size)])
    if (x > zeros).all():
        return 1
    return 0

def reLU(x):
    x_length = len(x)
    for i in range(x_length):
        if (x[i]<0):
            x[i] = 0
    return x


# PreLU func and deriv
PreLU_Val = 0.01

PreLU = lambda x: np.maximum(np.multiply(PreLU_Val,x),x)

def PreLU_deriv(x):
    if ((np.multiply(PreLU_Val,x) > x).all()):
        return PreLU_Val
    return 1


# my function to read data
def loadtxt(filename):
    vectors = []

    with open(filename, 'r') as f:
        for line in f:
            vec = np.fromstring(line.strip(), sep=' ')
            vec /= 255.0
            vectors.append(vec)

    return vectors

# calculating y_hat
def forward(input,params,function):
    x = input[0]
    y = input[1]
    yVal = int(y)
    x.shape = (784,1)
    z1 = np.dot(params["weights1"], x) + params["bias1"]
    h1 = function(z1)
    z2 = np.dot(params["weights2"], h1) + params["bias2"]
    y_hat = softMax(z2)
    lossSum = -np.log(y_hat[yVal])
    return {"loss": lossSum, "yVal": yVal, "y_hat": softMax(z2), "z1": z1, "h1": h1, "z2": z2}

# finding the gradients
def backProp(input,softMax_deriv,hiddenFunc_deriv,forwardVals,params):
    weights2_grad = np.dot(softMax_deriv, forwardVals["h1"].T)
    bias2_grad = softMax_deriv
    bias1_temp1 = np.dot(params["weights2"].T,softMax_deriv)
    bias1_grad = bias1_temp1*hiddenFunc_deriv(forwardVals["h1"])
    weights1_grad = np.dot(bias1_grad, input[0].T)
    return {"w1_grad": weights1_grad, "b1_grad": bias1_grad, "w2_grad": weights2_grad, "b2_grad": bias2_grad}

# updating the params
def update_params(params, LR, gradients):
    params["weights1"] = params["weights1"] - LR["weights1"] * gradients["w1_grad"]
    params["bias1"] = params["bias1"] - LR["bias1"] * gradients["b1_grad"]
    params["weights2"] = params["weights2"] - LR["weights2"] * gradients["w2_grad"]
    params["bias2"] = params["bias2"] - LR["bias2"] * gradients["b2_grad"]
    return params

# testing machine on validation set
def test_on_validation_Inputs(params, hiddenFunc, validationInputs):
    sum_loss = 0.0
    matches = 0.0
    size = len(validationInputs)
    for x in validationInputs:
        forwardVals = forward(x,params,hiddenFunc)
        sum_loss += forwardVals["loss"]
        if forwardVals["y_hat"].argmax() == x[1]:
            matches += 1
    acc = matches / size
    avg_loss = sum_loss / size
    return avg_loss, acc

# training the machine
def train(params, epochs, hiddenFunc, hiddenFuncDeriv, LR, actualTrainInputs, validationInputs):
    for i in range(epochs):
        lossPerTrain = 0.0
        sizeOfTrain = len(actualTrainInputs)
        shuffle(actualTrainInputs)
        for x in actualTrainInputs:
            forwardVals = forward(x, params, hiddenFunc)
            lossPerTrain += forwardVals["loss"]
            soft_deriv = (forwardVals["y_hat"])
            soft_deriv[forwardVals["yVal"]] -= 1
            gradients = backProp(x, soft_deriv, hiddenFuncDeriv, forwardVals, params)
            params = update_params(params, LR, gradients)

        validation_loss, acc = test_on_validation_Inputs(params, hiddenFunc, validationInputs)
        print "Epoch number: %d\nAverage train loss: %f\nAverage validation loss: %f\n" % (i, lossPerTrain / sizeOfTrain, validation_loss), "Level of accuracy: {}%\n".format(acc * 100)


# forward function for test_x (no loss calculation)
def test_forward(input, params, function):
    input.shape = (784, 1)
    z1 = np.dot(params["weights1"], input) + params["bias1"]
    h1 = function(z1)
    z2 = np.dot(params["weights2"], h1) + params["bias2"]
    return softMax(z2)


################################################################################################
################################# Program starts here ##########################################
################################################################################################

# reading train data
train_x = loadtxt("train_x")
train_y = np.loadtxt("train_y")
myTrainData = zip(train_x, train_y)
shuffle(myTrainData)

# splitting to train and validation
sizeOfTrain = len(train_x)
sizeOfValidation = sizeOfTrain/5
sizeOfActualTrain = sizeOfTrain - sizeOfValidation

actualTrainInputs = myTrainData[:sizeOfActualTrain:]
validationInputs = myTrainData[sizeOfActualTrain::]

# my hyper params
LR = {"weights1": 0.01, "bias1": 0.01, "weights2": 0.01, "bias2": 0.01}
activationFunctions = {"sigmoid": lambda x: 1/(1+np.exp(-x)),
                       "tanH": lambda x: np.tanh(x),
                       "reLU": reLU,
                       "PreLU": PreLU}

activationFuncDevs = {"sigmoid": lambda x: x*(1-x),
                      "tanH": lambda x: 4/np.power((np.exp(-x) + np.exp(x)), 2),
                      "reLU": reLU_Deriv,
                      "PreLU" : PreLU_deriv}
numOfEpochs = 30
currentFunc = activationFunctions["sigmoid"]
currentDeriv = activationFuncDevs["sigmoid"]

# training the set
train(params, numOfEpochs, currentFunc, currentDeriv, LR, actualTrainInputs, validationInputs)

# creating test_pred file
test_x = loadtxt("test_x")

f = open("test.pred", "w")
for x in test_x:
    y_hat = test_forward(x, params, currentFunc)
    f.write(str(y_hat.argmax()) + "\n")
f.close()
