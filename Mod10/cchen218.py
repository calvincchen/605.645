import sys
import random
import math
from copy import deepcopy

clean_data = {
    "plains": [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, "plains"]
    ],
    "forest": [
        [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, "forest"],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, "forest"],
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, "forest"],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, "forest"]
    ],
    "hills": [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, "hills"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, "hills"],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, "hills"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, "hills"]
    ],
    "swamp": [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, "swamp"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, "swamp"]        
    ]
}

def blur(data):
    def apply_noise(value):
        if value < 0.5:
            v = random.gauss(0.10, 0.05)
            if v < 0.0:
                return 0.0
            if v > 0.75:
                return 0.75
            return v
        else:
            v = random.gauss(0.90, 0.10)
            if v < 0.25:
                return 0.25
            if v > 1.00:
                return 1.00
            return v
    noisy_readings = [apply_noise(v) for v in data[0:-1]]
    return noisy_readings + [data[-1]]

def generate_data(clean_data, total_per_label):
    labels = len(clean_data.keys())
    def one_hot(n, i):
        result = [0.0] * n
        result[i] = 1.0
        return result

    data = []
    for i, label in enumerate(clean_data.keys()):
        for _ in range(total_per_label):
            datum = blur(random.choice(clean_data[label]))
            xs = datum[0:-1]
            ys = one_hot(labels, i)
            data.append((xs, ys))
    random.shuffle(data)
    return data

def learn_model(data, n_hidden, debug=False):
    '''
    Main function that returns the model for the given data set
    :param data: input data that is list of list of tuples
    :param n_hidden: number of hidden nodes
    :param debug: toggle for printing out debug statements
    :return: theta_h, theta_f. Theta values for the hidden and final nodes respectively
    '''
    alpha = .01
    epsilon = .01 # some experimentation showed that this generated a correct enough model in reasonable time (~30 min)
    n_final = 4
    previous_error = 100
    iterations = 0

    # generate the initial theta values for hidden and final

    #generating hidden theta values
    size_x = len(data[0][0])
    theta_h = []
    for _ in range(n_hidden):
        temp = []
        # add bias here
        temp.append(.01)
        for _ in range(size_x):
            temp.append(random.uniform(-1, 1))
        theta_h.append(temp)
    # theta h should be a list of lists, 4x17

    theta_final = []
    for _ in range(n_final):
        temp = []
        # add bias here
        temp.append(0.01)
        for _ in range(len(theta_h)):
            temp.append(random.uniform(-1, 1))
        theta_final.append(temp)
    # theta_final should be a list of lists, 4x5

    #main code
    while previous_error > epsilon:
        for d in data:
            if iterations % 1000 == 0 and debug:
                print(previous_error)
            if previous_error < epsilon:
                # program complete
                # return model
                return theta_h, theta_final
            else:
                x_values = d[0]
                expected_result = d[1]
                hidden_values, final_values = forward_step(x_values, theta_h, theta_final)

                current_error = calc_error(final_values, expected_result)
                if current_error > previous_error:
                    alpha /= 10
                else:
                    alpha = .01

                previous_error = current_error

                new_theta_h, new_theta_f = back_propagate(expected_result, final_values, hidden_values, theta_h, theta_final, alpha, x_values)
                theta_h, theta_final = new_theta_h, new_theta_f
                iterations += 1

    return theta_h, theta_final

def calc_n(input_list, theta_list):
    '''
    Calculates the value of a single node, hidden or final, depending on inputs
    :param input_list: parent node values
    :param theta_list: theta values for parent nodes to current node being evaluated
    :return: res. The resultant value of the current node
    '''
    # assume 1 to 1 mapping of input list to theta list, with extra theta for the bias
    # 1. calculate z
    # 2. apply logistic regression
    z = theta_list[0]
    for i in range(len(input_list)):
        z += input_list[i] * theta_list[i+1]

    res = 1 / (1 + math.exp(-z))
    return res


def forward_step(x_values, theta_h, theta_final):
    '''
    Calculates one iteration of forward stepping, from input x nodes to final, including hidden layer
    :param x_values: all values of x stored as list
    :param theta_h: all values of theta_h store as list of lists, where each inner list represents all theta values pointing to a single hidden node
    :param theta_final: all values of theta_f store as list of lists, where each inner list represents all theta values pointing to a single final node
    :return: h_values, f_values. Values for the hidden and final nodes, stored as lists
    '''
    # inputs are the given x (List), hidden theta (List of size x + 1 for bias) and final theta (List of size theta_h + 1 for bias)
    # calculates the forward steps and returns a tuple containing list of hidden_values and list of final_values (should both be size 4)

    # calculate the hidden layer
    h_values = []
    for h in theta_h: # should have 4
        h_values.append(calc_n(x_values, h))

    f_values = []
    for f in theta_final:
        f_values.append(calc_n(h_values, f))


    return h_values, f_values

def calc_delta_output(actual, expected):
    '''
    Calculates the delta value of a single final node
    :param actual: actual value of node
    :param expected: expected value of final node
    :return: delta of final node
    '''
    delta = actual * (1-actual) * (expected - actual)
    return delta


def calc_delta_hidden(actual, delta_children, theta):
    '''
    Calculate the delta value of a single hidden node
    :param actual: actual value of the node
    :param delta_children: list of delta values of all children (final nodes that it points to) nodes
    :param theta: list of theta values of all children nodes
    :return: delta of hidden node
    '''
    # theta and children should be of equal lengths
    y_hat = 0

    for c, t in zip(delta_children, theta):
        y_hat += c * t
    delta = y_hat * actual * (1-actual)

    return delta

def calc_error(raw_f_values, expected_f_values):
    '''
    Calculates the error of the current model
    :param raw_f_values: list of actual values of final nodes
    :param expected_f_values: list of expected values of final nodes
    :return: error as calculated using multi-class classifcation entropy
    '''
    #calculates error for the whole map?

    # convert f_values to 1's and 0s, 1 where it is highest, 0 everything else
    # converts first max value found...hopefully not an issue
    max_idx = raw_f_values.index(max(raw_f_values))
    actual_f_values = [0] * len(raw_f_values)
    actual_f_values[max_idx] = 1

    error = 0
    for idx in range(len(raw_f_values)):
        if not expected_f_values[idx]:
            p = 1-raw_f_values[idx]
        else:
            p = raw_f_values[idx]
        error += (int(actual_f_values[idx] == expected_f_values[idx])) * math.log(p)
    error *= -1
    return error

def extract_column(data, col):
    '''
    Returns a column slice at index col from input data
    :param data: data to be sliced
    :param col: column idx at which to slice
    :return: the sliced column
    '''
    return [row[col] for row in data]

def update_theta(delta_layer, parent_values, alpha, theta_layer):
    '''
    Updates all incoming theta values of an entire layer
    :param delta_layer: list of delta values for that layer
    :param parent_values: list of all incoming parent values
    :param alpha: adjustment variable
    :param theta_layer: list of all incoming parent thetas
    :return: list of new incomingtheta values
    '''
    # update theta is per layer (hidden or final)
    # theta_layer is all theta going into that layer. should be len(delta_layer) * (len(parent_values) + 1)

    new_theta = []
    for d, theta in zip(delta_layer, theta_layer):
        for t_idx in range(len(theta)):
            if t_idx == 0:
                #bias
                new_theta.append(theta[t_idx] + alpha * d)
            else:
                new_theta.append(theta[t_idx] + alpha * d * parent_values[t_idx - 1])

    return new_theta


def back_propagate(expected_final, final_values, hidden_values, theta_h, theta_final, alpha, x_values):
    '''
    Handles back propagation step of gradient descent
    :param expected_final: list of expected final values
    :param final_values: list of actual final values
    :param hidden_values: list of all hidden node values
    :param theta_h: list of incoming hidden theta
    :param theta_final: list of incoming final theta
    :param alpha: adjustment varaible
    :param x_values: list of all initial values
    :return:
    '''
    # calculates back prop
    # should return tuple of updated theta values for both hidden layer and final layer

    # 1. calculate delta for output layer. should be same length as final_values
    # 2. calculate delta for hidden layer
    # 3. update the theta and returns them

    delta_final = []
    for e, a in zip(expected_final, final_values):
        delta_final.append(calc_delta_output(a, e))

    delta_hidden = []
    for i in range(len(hidden_values)):
        children_theta = extract_column(theta_final, i + 1) # i+1 is to ignore bias
        delta_hidden.append(calc_delta_hidden(hidden_values[i], delta_final, children_theta))


    new_theta_h = update_theta(delta_hidden, x_values, alpha, theta_h)
    new_theta_final = update_theta(delta_final, hidden_values, alpha, theta_final)


    # new_theta_? are currently just lists. need to convert to list of lists
    # new theta h should be new list every len(x_values) + 1
    # new theta f should be new list every len(h_values) + 1
    new_theta_h = [new_theta_h[i:i+(len(x_values) + 1)] for i in range(0, len(new_theta_h), (len(x_values) + 1))]
    new_theta_final = [new_theta_final[i:i + (len(hidden_values) + 1)] for i in range(0, len(new_theta_final), (len(hidden_values) + 1))]

    return new_theta_h, new_theta_final

def apply_model(model, data, labeled=False):
    '''
    Implemented as described in the program instructions
    :param model: tuple containing two list of lists, one for hidden thetas and one for final thetas
    :param data: data to be tested
    :param labeled: whether or no the data is labeled
    :return: dependent on labeled field.
    '''
    # model is a tuple, where first element is list of lists for hidden layer theta, and second element is list of list for final layer theta

    if not labeled:
        res = []
        for d in data:
            # should be a list of input x values
            x_values = d[0]
            expected = d[1]
            h_values, f_values = forward_step(x_values, model[0], model[1])
            temp = []

            max_idx = f_values.index(max(f_values))
            actual_f_values = [0] * len(f_values)
            actual_f_values[max_idx] = 1

            for a, p in zip(actual_f_values, f_values):
                temp.append((a, p))
            res.append(temp)
        return res

    else:
        res = []
        for d in data:
            x_values = d[0]
            expected = d[1]

            h_values, f_values = forward_step(x_values, model[0], model[1])
            temp = []

            max_idx = f_values.index(max(f_values))
            actual_f_values = [0] * len(f_values)
            actual_f_values[max_idx] = 1

            for a, e in zip(actual_f_values, expected):
                temp.append((e, a))
            res.append(temp)
        return res

    return []

def evaluate_results(results):
    '''
    Implemented per program instructions. Only works with labeled results.
    :param results: list of list of tuples, where first entry is actual and second entry is expected value
    :return:
    '''
    # results is a list of list of tuples
    count = len(results)
    misclassified = 0

    for row in results:
        for e in row:
            if e[0] != e[1]:
                misclassified +=1
                break

    return misclassified / count

if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'


    train_data = generate_data(clean_data, 100)

    model = learn_model(train_data, 4, True)


    #model = ([[-3.5042668505623613, -1.8406264993717834, -0.8892232320698489, -1.351189131456613, -0.6426792583972651, -1.842973964407186, -3.9466964338644535, -3.159326915722514, -2.6712930421680747, 4.123850492426731, 3.1077962469644254, 3.137303285294535, 4.583153664522155, 0.6313114195089774, -0.29431865691335246, 0.6888996079907098, -0.28852227380291706], [0.3629177071853272, -0.6431449492373944, -1.0616390812941474, -0.6327609288850424, -0.9549519479304966, -1.4222948922689, -1.265729537337137, -1.5828153577404478, -1.1915706690164802, -1.2194374930404175, -0.8960703204405772, -1.3006320790039034, -0.5893388307720033, 1.921612789732685, 3.51601797146691, 3.5054449365922804, 1.434229660917888], [0.6751921847957716, 0.785670569956759, 0.31572925176564814, 0.7695418796452184, 1.2832749796673315, 1.1498443955562796, 1.9622360220527728, 2.6437960744360827, 0.744181162012053, -0.9620054454377966, -0.2155451718681472, -0.06407006206771712, -0.9358850811458639, -0.04398613145991215, -0.9631221497926884, -0.8741915965782963, -0.7125988672466512], [0.47683221396149683, 0.10594691960740305, 0.13284786326719633, -0.7465285307047833, 0.9228530905562012, -3.833340644863147, -2.709446329820147, -3.707990441417766, -3.7676340459165094, -0.9354234292751491, -4.422842064907279, -4.44926227011031, -0.7948566291098405, 4.2037102522472445, 0.7741123897810395, 1.1751229092803102, 4.169544375540622]], [[-2.1965312477360284, -9.899592826086007, 1.9971222874563412, -1.410501189775378, 5.830969783475513], [1.7324502843715046, -1.1822237969162828, -7.608778762208425, 4.24930891838724, -2.2942992081817897], [-2.0461457576738282, 1.435341412268374, 7.504135914618161, -3.9217078846634243, -11.18015751435779], [-7.420169483014364, 9.098183946863434, -2.674143197696148, -6.792584251666426, 6.706885911950078]])

    test_data = generate_data(clean_data, 100)
    # was false before, but not possible if running error_rate on result
    results = apply_model(model, test_data, True)

    for result in results[0:10]:
        print(result)

    error_rate = evaluate_results(results)
    print(f"The error rate is {error_rate}")



    # some test examples from self check
'''
    x = [0.52, -0.97]
    t_h = [[.01, .26, -.42], [-0.05, 0.78, 0.19], [0.42, -0.23, 0.37]]
    t_f = [[0.2, 0.61, 0.12, -0.9], [0.3, 0.28, -0.34, 0.1]]
    #print(calc_n(x, t_h[0]))
    hidden_values, final_values = forward_step(x, t_h, t_f)
    new_theta_h, new_theta_f = back_propagate([1, 0], final_values, hidden_values, t_h, t_f, .01, x, hidden_values)
'''
    #hidden_values, final_values = forward_step(x, new_theta_h, new_theta_f)
    #new_theta_h, new_theta_f = back_propagate([1, 0], final_values, hidden_values, new_theta_h, new_theta_f, .01, x, hidden_values)

    #update_theta(delta_hidden, x, .01, t_h)
    #print(new_theta_h)
    #print(new_theta_f)
