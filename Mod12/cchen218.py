import sys
import csv
import random
from copy import deepcopy
import numpy as np


##
## I'm leaving the shape of your data to you.
## You may want to stick with our List of Lists format or
## you may want to change to a List of Dicts.
## You can use your function from last week.
## 
def read_data(filename):
    '''
    Reads the data from the file into a list of dicts
    :param filename: string containing name of file in parent directory
    :return: data
    '''
    #features = ['edible', 'cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor', 'gill-atttachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', ]

    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = []
        for row in csv_reader:
            #data.append([v for v in row])
            temp = {}
            count = 0
            for v in row:
                temp[count] = v
                count += 1
            data.append(temp)
    # first element is e or p, rest is feature
    return data


def initialize_attributes():
    '''
    Creates dictionary of attributes per dataset info
    :return: dictionary
    '''
    # all attributes can be ?

    attributes = {}

    # cap-shape: bell=b,conical=c,convex=x,flat=f,knobbed=k,sunken=s
    attributes[1] = ['b', 'c', 'x', 'f', 'k', 's', '?']
    # cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
    attributes[2] = ['f', 'g', 'y', 's', '?']
    # cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
    attributes[3] = ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y', '?']
    # bruises?: bruises = t, no = f
    attributes[4] = ['t', 'f', '?']
    # odor: almond = a, anise = l, creosote = c, fishy = y, foul = f, musty = m, none = n, pungent = p, spicy = s
    attributes[5] = ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's', '?']
    # gill-attachment: attached = a, descending = d, free = f, notched = n
    attributes[6] = ['a', 'd', 'f', 'n', '?']
    # gill-spacing: close = c, crowded = w, distant = d
    attributes[7] = ['c', 'w', 'd', '?']
    # gill-size: broad = b, narrow = n
    attributes[8] = ['b', 'n', '?']
    # gill-color: black = k, brown = n, buff = b, chocolate = h, gray = g, green = r, orange = o, pink = p, purple = u, red = e,white = w, yellow = y
    attributes[9] = ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y', '?']
    # stalk-shape: enlarging = e, tapering = t
    attributes[10] = ['e', 't', '?']
    # stalk-root: bulbous = b, club = c, cup = u, equal = e, rhizomorphs = z, rooted = r, missing =?
    attributes[11] = ['b', 'c', 'u', 'e', 'z', 'r', '?']
    # stalk-surface: above - ring: fibrous = f, scaly = y, silky = k, smooth = s
    attributes[12] = ['f', 'y', 'k', 's', '?']
    # stalk-surface: below - ring: fibrous = f, scaly = y, silky = k, smooth = s
    attributes[13] = ['f', 'y', 'k', 's', '?']
    # stalk-color-above-ring: brown = n, buff = b, cinnamon = c, gray = g, orange = o,pink = p, red = e, white = w, yellow = y
    attributes[14] = ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y', '?']
    # stalk-color-below-ring: brown = n, buff = b, cinnamon = c, gray = g, orange = o, pink = p, red = e, white = w, yellow = y
    attributes[15] = ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y', '?']
    # veil-type: partial = p, universal = u
    attributes[16] = ['p', 'u', '?']
    # veil-color: brown = n, orange = o, white = w, yellow = y
    attributes[17] = ['n', 'o', 'w', 'y', '?']
    # ring-number: none = n, one = o, two = t
    attributes[18] = ['n', 'o', 't', '?']
    # ring-type: cobwebby = c, evanescent = e, flaring = f, large = l, none = n, pendant = p, sheathing = s, zone = z
    attributes[19] = ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z', '?']
    # spore-print-color: black = k, brown = n, buff = b, chocolate = h, green = r, orange = o, purple = u, white = w, yellow = y
    attributes[20] = ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y', '?']
    # population: abundant = a, clustered = c, numerous = n, scattered = s, several = v, solitary = y
    attributes[21] = ['a', 'c', 'n', 's', 'v', 'y', '?']
    # habitat: grasses = g, leaves = l, meadows = m, paths = p, urban = u, waste = w, woods = d
    attributes[22] = ['g', 'l', 'm', 'p', 'u', 'w', 'd', '?']

    return attributes


def initialize_attributes_dict():
    '''
    Creates dictionary to obtain feature counts per dataset info
    :return: dictionary
    '''
    # all attributes can be ?

    attributes_dict = initialize_attributes()

    for key, features in attributes_dict.items():
        dictOfWords = {i: 1 for i in features}
        attributes_dict[key] = dictOfWords

    return attributes_dict


def train(training_data):
    """
    takes the training data and returns the probabilities need for NBC.
    """

    # The probabilities that we need are overall P(e) and P(p), and P(feature|e/p). Assume probabilities are independent for naive bayes

    total_count = len(training_data)
    e_count = 0
    p_count = 0

    given_e_probabilities = initialize_attributes_dict()
    given_p_probabilities = initialize_attributes_dict()

    for d in training_data:
        # getting counts
        if d[0] == 'e':
            e_count += 1
            for idx in range(1, len(d)):
                given_e_probabilities[idx][d[idx]] += 1
        else:
            p_count += 1
            for idx in range(1, len(d)):
                given_p_probabilities[idx][d[idx]] += 1

    # getting probabilities by dividing class label by feature counts
    for key, features in given_e_probabilities.items():
        probabilities = {i: features[i] / e_count for i in features}
        given_e_probabilities[key] = probabilities

    for key, features in given_p_probabilities.items():
        probabilities = {i: features[i] / p_count for i in features}
        given_p_probabilities[key] = probabilities

    P_e = e_count/total_count
    P_p = p_count/total_count

    # need to return P(e), P(p), given_e_probabilities, given_p_probabilities
    return (P_e, P_p, given_e_probabilities, given_p_probabilities)


def classify(probabilities, data):
    """
    Takes the probabilities needed for NBC, applies them to the data, and
    return a List of classifications.
    """
    res = []

    P_e = probabilities[0]
    P_p = probabilities[1]
    given_e_probabilities = probabilities[2]
    given_p_probabilities = probabilities[3]

    for d in data:
        # generate initial probabilities
        probability_e = P_e
        probability_p = P_p
        for feature_idx in range(1, len(d)):
            probability_e *= given_e_probabilities[feature_idx][d[feature_idx]]
            probability_p *= given_p_probabilities[feature_idx][d[feature_idx]]

        # normalize the probabilities to equal 1 based on weights
        normalize_e = probability_e * P_e / (probability_e * P_e + probability_p * P_p)
        normalize_p = probability_p * P_p / (probability_e * P_e + probability_p * P_p)

        res.append({"e": normalize_e, "p": normalize_p})

    return res


def evaluate(actual, predicted):
    """
    takes a List of actual labels and a List of predicted labels
    and returns the error rate.
    """

    # note that predicted is a dictionary of probabilities.
    # actual is all of the test data?

    if len(actual) != len(predicted):
        print("Error mismatch")
        return
    else:
        total = len(actual)
        count = 0
        for i, j in zip(actual, predicted):
            if i[0] != max(j, key=j.get):
                count += 1
        return count / total



def generate_folds(data, num_chunks):
    '''
    Generates num_chunks number of folds of given data
    :param data: input data
    :param num_chunks: number of folds
    :return: List of lists
    '''
    # n is number of chunks

    new_data = deepcopy(data)
    random.shuffle(new_data)

    return np.array_split(new_data, num_chunks)


def cross_validate(data):
    ##
    ## You can use your function from last week.
    ##
    ## combine train, classify and evaluate
    ## to perform 10 fold cross validation, print out the error rate for
    ## each fold and print the final, average error rate.

    folds = generate_folds(data, 10)
    all_error_rate = []
    for i in range(10):
        test_data = folds[i]
        train_data = []
        for j in range(10):
            if i != j:
                train_data.extend(folds[j])

        tree = train(train_data)
        predicted_results = classify(tree, test_data)

        error_rate = evaluate(test_data, predicted_results)
        all_error_rate.append(error_rate)

        print("The error rate with fold " + str(i + 1) + " as the test set is: " + str(error_rate))

    average_error = sum(all_error_rate) / len(all_error_rate)

    print("The average error rate across all folds is " + str(average_error))

    pass

if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'

    data = read_data('agaricus-lepiota.data')
    cross_validate(data)