import sys
import csv
import random
import numpy as np
from math import exp, log, floor
import matplotlib.pyplot as plt


"""
Notes:
    data is a list of lists
        1st n-1 items in each row are x values
        last value is they y value
        split into separate arrays
    length of theta should be n-1 + 1 (for x_0)
        x_0 theta typically the first one?
    length of y_hat should be the number of lists in data
    for calculating derivatives, remember to do x_0 first, then approach other theta
        will need adjust by 1 somewhere
"""


def read_data(filename):
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = []
        for row in csv_reader:
            data.append([float(v) for v in row])
    return data

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
    epsilon = .0000007
    alpha = .1
    return gradient_descent(data, alpha, epsilon, 'linear', debug)
    


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
    epsilon = .0000001
    alpha = .1
    return gradient_descent(data, alpha, epsilon, 'log', debug)


def apply_linear_regression(model, xs):
    """
    model is the parameters of a linear regression model and the xs are the inputs
    to the model not including x_0 = 1.

    returns the predicted y based on the model and xs.
    """
    res = model[0] # y intercept
    for idx in range(len(xs)):
        res += xs[idx] * model[idx + 1]
    return res

    

def apply_logistic_regression(model, xs):
    """
    model is the parameters of a logistic regression model and the xs are the inputs
    to the model not including x_0 = 1.

    returns the predicted y based on the model and xs.
    """

    '''
    res = 1 / (1 + exp(-model[0])) # y intercept
    for idx in range(len(xs)):
        res += 1 / (1 + exp(-xs[idx] * model[idx + 1]))
    '''
    z = model[0]
    for idx in range(len(xs)):
        z += xs[idx] * model[idx + 1]
    res = 1 / (1 + exp(-z))

    return res

def gradient_descent(data, alpha, epsilon, reg_type, debug):
    '''
    Function to calculate gradient descent. Adapated from pseudocode provided in Mod8
    :param data: List of Lists containing the input csv data, where the last column is the actual y values
    :param alpha: adaptive alpha to use for convergence
    :param epsilon: max delta between iterations
    :param reg_type: string representing type of regression to use. Currently only supports 'linear' and 'log'
    :param debug: if true, prints the current error at every step (missing the final step)
    :return theta: the model. List of theta's
    '''
    data = np.array(data)

    xs = data[:,:-1]
    ys = data[:,-1]

    theta = [random.randint(-100, 100)/100 for _ in range(len(xs[0])+1)]
    previous_error = 0
    y_hat = calculate_y_hat(theta, xs, reg_type)
    current_error = calculate_error(ys, y_hat, reg_type)
    while abs(current_error - previous_error) > epsilon:
        if debug:
            print("Current error: " + str(current_error))
        new_theta = []
        theta_derivatives = derivative(theta, xs, ys, y_hat)
        for i in range(len(theta)):
            new_t = theta[i] - alpha * theta_derivatives[i]
            new_theta.append(new_t)

        theta = new_theta
        previous_error = current_error
        y_hat = calculate_y_hat(theta, xs, reg_type)
        current_error = calculate_error(ys, y_hat, reg_type)
        if previous_error < current_error:
            alpha /= 10
        else:
            alpha = .1

    return theta

def calculate_y_hat(theta, xs, reg_type):
    '''
    Helper function that determines which type of y_hat calculation to use
    :param theta: List of current theta values, ordered by index
    :param xs: List of Lists where inner lists represent all x values for an expression, and outer list is all expressions
    :param reg_type: string representing type of regression to use. Currently only supports 'linear' and 'log'
    '''

    if reg_type == 'linear':
        return calculate_linear_y_hat(theta, xs)
    elif reg_type == 'log':
        return calculate_log_y_hat(theta, xs)
    else:
        print("Expected 'linear' or 'log' as regression type. Invalid regression type provided")
        sys.exit()
        
        #return calculate_log_y_hat(theta, xs)

def calculate_error(theta, xs, reg_type):
    '''
    Helper function that determines which type of error calculation to use
    :param theta: List of current theta values, ordered by index
    :param xs: List of Lists where inner lists represent all x values for an expression, and outer list is all expressions
    :param reg_type: string representing type of regression to use. Currently only supports 'linear' and 'log'
    :return: appropriate function to calculate error
    '''
    if reg_type == 'linear':
        return calculate_linear_error(theta, xs)
    else:
        return calculate_log_error(theta, xs)
        
        #return calculate_log_y_hat(theta, xs)

def calculate_linear_y_hat(theta, xs):
    '''
    Function to calculate the y_hat of a linear function given theta and xs
    :param theta: List of current theta values, ordered by index
    :param xs: List of Lists of x's, where inner lists represent expressions. Note that inner lists trail theta by 1 index b/c x_0
    :return: list of y_hat representing y_hat of each expression (inner list of x's)
    '''

    y_hat = []
    for row in xs:
        temp = 0
        for col_idx in range(len(row)):
            temp += row[col_idx] * theta[col_idx+1]
        temp += theta[0]
        y_hat.append(temp)

    return y_hat

def calculate_log_y_hat(theta, xs):
    '''
    Function to calculate the y_hat of a log function given theta and xs
    :param theta: List of current theta values, ordered by index
    :param xs: List of Lists of x's, where inner lists represent expressions. Note that inner lists trail theta by 1 index b/c x_0
    :return: list of y_hat representing y_hat of each expression (inner list of x's)
    '''
    y_hat = []
    for row in xs:
        z = 0
        for col_idx in range(len(row)):
            z += row[col_idx] * theta[col_idx+1]
        z += theta[0]
        z = 1 / (1 + exp(-z))
        y_hat.append(z)

    #y_hat[x] is the expected y value at row x
    return y_hat


def calculate_linear_error(ys, y_hat):
    '''
    Function to calculate the error of a linear function
    :param ys: List of actual y values
    :param y_hat: List of expected y values
    :return: mean error of the expected and actual y values
    '''
    res = 0
    for row_idx in range(len(ys)):
        res += (y_hat[row_idx] - ys[row_idx])**2
    res = res / 2 / len(ys)
    return res

def calculate_log_error(ys, y_hat):
    '''
    Function to calculate the error of a log function
    :param ys: List of actual y values
    :param y_hat: List of expected y values
    :return: mean error of the expected and actual y values
    '''

    res = 0
    for row_idx in range(len(ys)):
        y_i = ys[row_idx]
        y_hat_i = y_hat[row_idx]
        # catches cases where y_i = 0 or 1 and don't need to evaluate log statement
        try:
            res += y_i * log(y_hat_i) + (1-y_i) * (log(1-y_hat_i))
        except:
            if y_i == 0:
                res += (1-y_i) * (log(1-y_hat_i))
            else:
                res += y_i * log(y_hat_i)
    res = -res / len(ys)

    return res

def classification_metrics(TP, FP, TN, FN):
    '''
    Unused function, error metrics for classification
    :param TP: True positive amount
    :param FP: False positive amount
    :param TN: True negative amount
    :param FN: False negative amount
    :return:
    '''
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    error = (FP + FN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return accuracy, error, precision, recall

def cross_validation(data, num_blocks):
    '''
    Returns a list of list of randomized data, split into equal block sizes
    :param data:
    :param num_blocks:
    :return:
    '''
    random.shuffle(data)
    blocks = []

    start_idx = 0
    block_length = len(data) // num_blocks

    for i in range(num_blocks):
        blocks.append(data[start_idx: start_idx + block_length])
        start_idx += block_length

    # capture remainder of data
    blocks[-1].extend(data[start_idx:])
    return blocks



def derivative(theta, xs, ys, y_hat):
    '''
    Function to return a list of derivatives for each theta value
    :param theta: list of theta values (should be of size xs[0] + 1)
    :param xs: List of Lists of x's, where inner lists represent expressions. Note that inner lists trail theta by 1 index b/c x_0
    :param ys: List of actual y values
    :param y_hat: List of expected y values
    :return list of derivatives of one per theta
    '''
    # should have one derivative per column
    # return array of derivatives?
    # remember x_0 = 1
    res = []
    n = len(ys)

    # need two loops. outer is thetas, inner is xs
    for theta_idx in range(len(theta)):
        if theta_idx == 0:
            # do x_0 stuff
            x_ij = [1] * len(ys)
        else:
            # gets splice of all x_i's
            x_ij = xs[:,theta_idx-1]
        temp = 0
        for y_idx in range(len(ys)):
            temp += (y_hat[y_idx] - ys[y_idx]) * x_ij[y_idx]
        temp = temp/n
        res.append(temp)

    return res


def bias(data):
    '''
    Part 2 of the problem. plots the bias-variance of the model to test for convergence
    :param data: input data
    :return:
    '''
    train_error = []
    test_error = []
    p = []

    data_blocks = cross_validation(data, 5)
    train_data = data_blocks[0] + data_blocks[1] + data_blocks[2] + data_blocks[3]
    test_data = data_blocks[4]

    for proportion in range(5, 105, 5):  # determine step size here
        temp_train_data = train_data[:floor(proportion / 100 * len(train_data))]
        model = learn_logistic_regression(temp_train_data)
        '''
        temp_train_error = 0
        temp_test_error = 0
        for point in temp_train_data:
            temp_train_error += apply_logistic_regression(model, point[:-1]) - point[-1]
        for point in test_data:
            temp_test_error += apply_logistic_regression(model, point[:-1]) - point[-1]
        
        # normalizing the error
        temp_train_error /= len(temp_train_data)
        temp_test_error /= len(test_data)
        
        train_error += [temp_train_error]
        test_error += [temp_test_error]
        '''
        train_error += [MSE(temp_train_data, model)] # should this be train_data or temp_train_data
        test_error += [MSE(test_data, model)]
        p += [proportion]
    #print(train_error)
    #print(test_error)
    #print(p)
    print("Plotting for convergence...")
    plot(p, train_error, test_error, 1, True, "Percent Training Data Used", "Error")
    return
    # plot the functions


def plot(x, train_y, test_y, degree=1, log_fit=False, xaxis='x axis', yaxis='y axis'):
    '''
    Creates a best fit line for the input data and graphs it
    :param x: values for x axis (could be parameters)
    :param train_y: y values from training data
    :param test_y: y values from testing data
    :param degree: polynomial degree
    :param log_fit: True if we want to use a log fit
    :param xaxis: label for x axis
    :param yaxis: label for y axis
    :return: trendlines for training and test data
    '''
    # plot points
    #plt.plot(x, train_y, 'bo', label="Training error")
    #plt.plot(x, test_y, 'ro', label="Testing error")
    #plt.legend()

    if log_fit:
        trend1 = np.polyfit(np.log(np.array(x)), train_y, degree)
        trend2 = np.polyfit(np.log(np.array(x)), test_y, degree)

        trendpoly1 = np.poly1d(trend1)
        trendpoly2 = np.poly1d(trend2)

        plt.plot(x, trendpoly1(np.log(np.array(x))), label="Training error")
        plt.plot(x, trendpoly2(np.log(np.array(x))), label="Testing error")
    else:
        trend1 = np.polyfit(x, train_y, degree)
        trend2 = np.polyfit(x, test_y, degree)

        trendpoly1 = np.poly1d(trend1)
        trendpoly2 = np.poly1d(trend2)

        plt.plot(x, trendpoly1(x), label="Training error")
        plt.plot(x, trendpoly2(x), label="Testing error")

    plt.legend()
    # plot the equations


    plt.ylabel(yaxis)
    plt.xlabel(xaxis)

    plt.show()

    # inflection point is -b/2a
    # trend2 is an array of the coefficients
    return trend1, trend2

# check if there is low or high bias visually by seeing if the error converge at 100%


def MSE(data, model):
    '''
    Calculates the mean squared error of the model given representative data
    :param data: data set, with each row representing a tuple of associated x and y values
    :param model: model
    :return: mean squared erorr
    '''
    sum_error_squared = 0

    for point in data:
        actual = apply_logistic_regression(model, point[:-1])
        expected = point[-1]
        error = expected - actual
        sum_error_squared += error * error

    mean_squared_error = sum_error_squared / len(data)
    return mean_squared_error


def find_inflection_point(data):
    '''
    Finds the inflection point/threshold for the given data for part 3
    :param data: input data
    :return: None
    '''
    # using functions defined in previous questions

    data_blocks = cross_validation(data, 5)
    train_data = data_blocks[0] + data_blocks[1] + data_blocks[2] + data_blocks[3]
    test_data = data_blocks[4]

    test_error = []
    train_error = []
    threshold = []
    threshold_range = [x * 0.1 for x in range(0, 11)]

    model = learn_logistic_regression(train_data)

    for t in threshold_range:
        temp_train_error = 0
        temp_test_error = 0
        for point in train_data:
            expected = point[-1]
            actual = 1 if apply_logistic_regression(model, point[:-1]) >= t else 0
            temp_train_error += int(not expected == actual) ** 2
        for point in test_data:
            expected = point[-1]
            actual = 1 if apply_logistic_regression(model, point[:-1]) >= t else 0
            temp_test_error += int(not expected == actual) ** 2

        train_error += [temp_train_error / len(train_data)]
        test_error += [temp_test_error / len(test_data)]
        threshold += [t]

    #print(train_error)
    #print(test_error)
    #print(threshold)
    # graph and find x value at minimum
    print("Plotting to find the best threshold value...")
    test_eq, trend_eq = plot(threshold, train_error, test_error, 2, False, "Threshold value", "Error")
    inflection = -trend_eq[1] / 2 / trend_eq[0]
    print("The best threshold value is: " + str(inflection))
    return


if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'
    
    
    data = read_data("logistic_regression.csv")
    model = learn_logistic_regression(data)
    error = MSE(data, model)
    print("Error using MSE: " + str(error))
    bias(data)
    find_inflection_point(data)
    '''
    logistic_regression_model = learn_logistic_regression(data, debug)
    print("logistic regression model: ", logistic_regression_model)
    for point in data[0:10]:
        print(point[-1], apply_logistic_regression(logistic_regression_model, point[:-1]))
    '''


    