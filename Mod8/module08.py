import sys
import csv
from random import randint

def read_data(filename):
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = []
        for row in csv_reader:
            data.append([float(v) for v in row])
    return data

def gradient_descent(xs, y, thetas, type):
    previous_error = 0.0
    current_error, yhat = calculate_error(thetas, xs, y)
    print(yhat)
    print(current_error)
    epsilon = 10**-7
    alpha = 0.1
    delta_error = current_error - previous_error
    new_thetas = thetas
    while abs(delta_error) >= epsilon:
        der = derivative(y, yhat, xs, new_thetas)
        new_thetas = update_thetas(new_thetas, der, alpha)
        previous_error = current_error
        current_error , yhat = calculate_error(new_thetas, xs, y)
        if current_error > previous_error:
            alpha = alpha / 10
        else:
            alpha = 0.1
        delta_error = current_error - previous_error
    return new_thetas

def update_thetas (thetas, der, alpha):
    output = []
    for i in range(len(thetas)):
        output.append(thetas[i] - der[i]*alpha)
    return output

def derivative (y, yhat, xs, thetas):
    row_count = len(y)
    total_der = []
    for i in range(row_count):
        y_delta = yhat[i] - y[i]
        row_der = [1]
        #for each x, add to the row's derivative
        for item in xs[i]:
            row_der.append(y_delta * item)
        total_der.append(row_der)
    sum_der = []
    for item in range(len(thetas)):
        sum_der.append(0)
    for i in range(row_count):
        for j in range(len(total_der[0])):
            sum_der[j] = sum_der[j] + total_der[i][j]
    for i in range(len(sum_der)):
        sum_der[i] = sum_der[i] / row_count
    return sum_der


    # divide by #x's
    num_x = len(xs[0]) + 1  # for x_0
    for item in sum_der:
        item/num_x
    return sum_der

def calculate_error(thetas, xs, y):
    row_count = len(y)
    total_estimate = 0
    yhat = []
    for i in range(row_count):
        row_xs = xs[i]
        row_y = y[i]
        row_yhat = 0
        for j in range(len(thetas)-1):
            row_yhat += thetas[j+1] * row_xs[j]
        row_yhat += thetas[0]
        yhat.append(row_yhat)
        row_estimate = row_yhat - row_y
        total_estimate += row_estimate**2

    total_estimate = total_estimate / row_count / 2
    return total_estimate , yhat


def split(data):
    row_size = len(data[0])
    y = []
    xs = []
    for row in data:
        last_element = row[row_size -1]
        y.append(last_element)
        xs.append(row[:-1])
    return xs, y

def initialize_thetas( num_thetas):
    row = []
    while len(row) < num_thetas + 1:
        row.append(randint(-100, 100)/100)
    return row
def learn_linear_regression(data, debug=False):
    """
    data is a list of lists where the last element. The outer list is all the data
    and each inner list is an observation or example for training. The last element of 
    each inner list is the target y to be learned. The remaining elements are the inputs,
    xs. The inner list does not include in x_0 = 1.

    This function uses gradient descent. If debug is True, then it will print out the
    the error as the function learns. The error should be steadily decreasing.

    returns the parameters of a linear regression model for the data.
    """
    xs, y = split(data)

    thetas = [0.24, -0.89, -0.53] #initialize_thetas(len(xs[0]))


    return gradient_descent(xs, y, thetas, "linear")


def learn_logistic_regression(data, debug=False):
    """
    data is a list of lists where the last element. The outer list is all the data
    and each inner list is an observation or example for training. The last element of 
    each inner list is the target y to be learned. The remaining elements are the inputs,
    xs. The inner list does not include in x_0 = 1.

    This function uses gradient descent. If debug is True, then it will print out the
    the error as the function learns. The error should be steadily decreasing.

    returns the parameters of a logistic regression model for the data.
    """
    pass

def apply_linear_regression(model, xs):
    """
    model is the parameters of a linear regression model and the xs are the inputs
    to the model not including x_0 = 1.

    returns the predicted y based on the model and xs.
    """
    res = model[0]  # y intercept
    for idx in range(len(xs)):
        res += xs[idx] * model[idx + 1]
    return res

def apply_logistic_regression(model, xs):
    """
    model is the parameters of a logistic regression model and the xs are the inputs
    to the model not including x_0 = 1.

    returns the predicted y based on the model and xs.
    """
    pass

if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'


    data = read_data("linear_regression.csv")
    linear_regression_model = learn_linear_regression(data, debug)
    print("linear regression model: ", linear_regression_model)
    for point in data[0:10]:
         print(point[-1], apply_linear_regression(linear_regression_model, point[:-1]))
    #
    #
    # data = read_data("logistic_regression.csv")
    # logistic_regression_model = learn_logistic_regression(data, debug)
    # print("logistic regression model: ", logistic_regression_model)
    # for point in data[0:10]:
    #     print(point[-1], apply_logistic_regression(logistic_regression_model, point[:-1]))