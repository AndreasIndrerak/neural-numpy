import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#This is a manually programmed neural network only using numpy
#The results vary to a vast degree, there are no changeable hyperparameters
#The test dataset is really way to small, the odds of overfitting are drastic
#Usually get a result = 80% accuracy, but this varies each time, all the way down to 60%
#There are no validation datasets
#Lots of changes could be made to improve performance,
#but the goal of this project is to showcase how a neural-network forward->backward propagation works

#42000 entries of 28*28 = 784 pixel images of handwritten numbers from 0->9
data = pd.read_csv('train.csv')

#Dataframe -> Numpy
data = np.array(data)
m, n = data.shape
#Not using random seed, but we could
np.random.shuffle(data)
test_size = 4000
#data_dev = data_test. Should be a larger subset. Should also be randomised parts
data_dev = data[0:test_size].T
y_dev = data_dev[0] #just 1000 numbers from 0 -> 9
x_dev = data_dev[1:n]
x_dev = x_dev / 255 #Data is grey pixels ranging from 0 -> 255, we normalise this to be between 0 and 1

#1000 = magic number, should be a variable
data_train = data[test_size:m].T
y_train = data_train[0]
x_train = data_train[1:n]
x_train = x_train / 255#add . after 255 if this doesn't act as a float division


#Initialising the parameters between -0.5 and 0.5
def init_params():
    w1 = np.random.rand(10, 784) - 0.5      #hidden x input
    b1 = np.random.rand(10, 1) - 0.5        #hidden x 1
    w2 = np.random.rand(10, 10) - 0.5       #hidden x output
    b2 = np.random.rand(10, 1) - 0.5        #output x 1
    return w1, b1, w2, b2

#relu activation function, removes negative numbers
def relu(z):
    return np.maximum(0, z)

def deriv_relu(z):
    return z > 0

#Transfroms logits into probability, used in final layer of classifier models
def softmax(logits):
    probability = np.exp(logits) / sum(np.exp(logits))
    return probability

    #Data from input to output
def forward_prop(w1, b1, w2, b2, x):
    preAcValue = w1.dot(x) + b1                 #linear transformation first layer
    activatedValue = relu(preAcValue)           #after activation
    rawLogits = w2.dot(activatedValue) + b2     #activated value used to form datapoints
    probabilities = softmax(rawLogits)          #datapoints turned into probabilities
    #tidligere z1, a1, z2, a2
    return preAcValue, activatedValue, rawLogits, probabilities

    #Geeks for geeks explanation of what one_hot encoding does:
    #One Hot Encoding is a method for converting categorical variables into a binary format.
    #It creates new columns for each category where 1 means the category is present and 0 means it is not. 
    # The primary purpose of One Hot Encoding is to ensure that categorical data can be 
    # effectively used in machine learning models.
def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

#Probegates error backwards through network to find loss gradient of each parameter, later used to update parameters
def back_prop(preAcValue, activatedValue,  probabilities,  w2, x, y):
    one_hot_y = one_hot(y)                      #Onehot encoding y which are the true labels
    dz2 = probabilities - one_hot_y             #Finds error between true and probable labels
    dw2 = 1 / m * dz2.dot(activatedValue.T)     #loss gradient with respect to weights of second layer
    db2 = 1 / m * np.sum(dz2)                   #loss gradient with respect to bias of second layer

    dz1 = w2.T.dot(dz2) * deriv_relu(preAcValue)#derviated relu multiplied with propagated error from output layer through w2, finds error propagated back to prior layer
    dw1 = 1 / m * dz1.dot(x.T)                  #loss gradient with respect to weights of first layer
    db1 = 1 / m * np.sum(dz1)                   #loss gradient with respect to bias of first layer
    return dw1, db1, dw2, db2

#updates parameters using gradients
def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2 ,b2

#Finds most probable value
def get_predictions(probabilities):
    return np.argmax(probabilities, 0)

#self-explanatory
def get_accuracy(predictions, y):
    acc = np.sum(predictions == y) / y.size
    return acc

#Makes predictions and returns it
def make_predictions(x, w1, b1, w2, b2):
    Z1, A1, Z2, A2 = forward_prop(w1, b1, w2, b2, x)
    predictions = get_predictions(A2)
    return predictions

#makes prediction and returns accuracy
def predict(x_dev, y_dev, w1, b1, w2, b2):
    dev_predict = make_predictions(x_dev, w1, b1, w2, b2)
    acc = get_accuracy(dev_predict, y_dev)
    return acc

#This is the gradient descent which combines all elements
def gradient_descent(x, y, iterations, alpha):
    acc_list = []
    predict_list = []
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        z1, a2, z2, a2 = forward_prop(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = back_prop(z1, a2, a2, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        
        if i % 10 == 0:
            #print("Iteration", i)
            prediction = get_predictions(a2)
            acc = get_accuracy(prediction, y)
            acc_list.append(acc)
            #print("Accuracy: ", Accuracy)
            current_acc = predict(x_dev, y_dev, w1, b1, w2, b2)
            predict_list.append(current_acc)
    return acc_list, predict_list

#plots the result, results may vary drastically
def plot_result(acc_list, predict_list):
    time = range(len(acc_list))
    time2 = range(len(predict_list))
    plt.plot(time, acc_list, marker='o', linestyle= '-', color='b')
    plt.plot(time2, predict_list, marker='o', linestyle= '-', color='r')
    plt.xlabel('Time (epochs or steps)')
    plt.ylabel('Accuracy')
    plt.title('ML Accuracy over Time')
    plt.grid(True)
    plt.show()

#trains model
def fit(x, y, iterations, alpha):
    acc_list, predict_list = gradient_descent(x, y, iterations, alpha)
    plot_result(acc_list, predict_list)

fit(x_train, y_train, 440, 0.1)