import sys
import csv
from copy import deepcopy
from collections import defaultdict
from math import log2
import random
import numpy as np


##
## I'm leaving the shape of your data to you.
## You may want to stick with our List of Lists format or
## you may want to change to a List of Dicts.
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
    print(data[0])
    return data

def calc_IG(data, attribute_no):
    '''
    Calculates the gain of the specified attribute
    :param data: input data
    :param attribute_no: attribute to evaluate
    :return: IG value
    '''
    # first find E(s) of the data
    total = len(data)

    master_count = defaultdict(int)
    for d in data:
        master_count[d[0]] += 1
    # hardcoded because we know it's always e or p
    master_e = E_s(master_count['e'], master_count['p'])


    # divide data for value in attribute, where attribute is the index
    # calc E for each one?
    total = len(data)

    data_split = defaultdict(list)

    for d in data:
        # d is a dictionary
        data_split[d[attribute_no]].append(d)

    e_vals = {}
    for k, v in data_split.items():
        e_count = 0
        p_count = 0
        for points in v:
            if points[0] == 'e':
                e_count += 1
            else:
                p_count += 1
        # now we have e, p, and total
        # can check to see if either of them are 0?
        if p_count == 0 or e_count == 0:
            e_vals[k] = 0
        else:
            e_vals[k] = E_s(e_count, p_count)

    # calc IG
    IG = 0
    for k, v in e_vals.items():
        IG += len(data_split[k]) / total * v

    IG = master_e - IG

    return IG

def create_subset(data, attribute, value):
    '''
    Creates a subset of data values where all points have the attribute value "value"
    :param data: input data
    :param attribute: attribute class
    :param value: attribute value
    :return: subset
    '''
    subset = []
    for d in data:
        if d[attribute] == value:
            subset.append(d)
    return subset

def E_s(p, n):
    '''
    Evaluates the E(s) given the provided formula
    :param p: number of positive
    :param n: number of negative
    :return: E(s)
    '''
    total = p + n
    return - (p/total) * log2(p/total) - (n/total) * log2(n/total)

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
    attributes[3] = ['n', 'b', 'c' , 'g' , 'r' , 'p' , 'u' , 'e' , 'w', 'y' , '?']
    # bruises?: bruises = t, no = f
    attributes[4] = ['t', 'f' , '?']
    # odor: almond = a, anise = l, creosote = c, fishy = y, foul = f, musty = m, none = n, pungent = p, spicy = s
    attributes[5] = ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's', '?']
    # gill - attachment: attached = a, descending = d, free = f, notched = n
    attributes[6] = ['a', 'd', 'f', 'n', '?']
    # gill - spacing: close = c, crowded = w, distant = d
    attributes[7] = ['c', 'w', 'd', '?']
    # gill - size: broad = b, narrow = n
    attributes[8] = ['b', 'n', '?']
    # gill - color: black = k, brown = n, buff = b, chocolate = h, gray = g, green = r, orange = o, pink = p, purple = u, red = e,white = w, yellow = y
    attributes[9] = ['k', 'n', 'b', 'h', 'g', 'r', 'o','p','u','e' ,'w', 'y','?']
    #stalk - shape: enlarging = e, tapering = t
    attributes[10] = ['e', 't', '?']
    # stalk - root: bulbous = b, club = c, cup = u, equal = e, rhizomorphs = z, rooted = r, missing =?
    attributes[11] = ['b', 'c', 'u', 'e' , 'z', 'r', '?']
    # stalk-surface: above - ring: fibrous = f, scaly = y, silky = k, smooth = s
    attributes[12] = ['f','y','k','s', '?']
    # stalk-surface: below - ring: fibrous = f, scaly = y, silky = k, smooth = s
    attributes[13] = ['f','y','k','s','?']
    # stalk-color: above - ring: brown = n, buff = b, cinnamon = c, gray = g, orange = o,pink = p, red = e, white = w, yellow = y
    attributes[14] = ['n','b','c','g','o','p','e','w','y','?']
    # stalk-color: below - ring: brown = n, buff = b, cinnamon = c, gray = g, orange = o, pink = p, red = e, white = w, yellow = y
    attributes[15] = ['n','b','c','g','o','p','e','w','y','?']
    # veil-type: partial = p, universal = u
    attributes[16] = ['p','u','?']
    # veil-color: brown = n, orange = o, white = w, yellow = y
    attributes[17] = ['n','o','w','y', '?']
    # ring-number: none = n, one = o, two = t
    attributes[18] = ['n','o', 't','?']
    # ring-type: cobwebby = c, evanescent = e, flaring = f, large = l, none = n, pendant = p, sheathing = s, zone = z
    attributes[19]= ['c','e','f','l', 'n', 'p','s','z','?']
    # spore-print - color: black = k, brown = n, buff = b, chocolate = h, green = r, orange = o, purple = u, white = w, yellow = y
    attributes[20] = ['k','n','b','h','r','o','u','w','y','?']
    # population: abundant = a, clustered = c, numerous = n, scattered = s, several = v, solitary = y
    attributes[21] = ['a','c','n', 's','v', 'y','?']
    # habitat: grasses = g, leaves = l, meadows = m, paths = p, urban = u, waste = w, woods = d
    attributes[22] = ['g','l','m','p','u','w','d','?']

    return attributes

def findBestAttribute(data, attributes):
    '''
    Finds the best attribute for the given data by taking largest IG
    :param data: input data
    :param attributes: available attribute list
    :return: name of best attribute
    '''
    m = 0
    best_attr = ''
    for a in attributes:
        score = calc_IG(data, a)
        #print(score)
        if score > m:
            m = score
            best_attr = a
    return best_attr


def ID3(data, attributes, default='p'):
    '''
    Adapted from Mod11 pseudocode
    :param data: input training data
    :param attributes: available attributes
    :param default: default value set to 'p' because it's better to have false negative than false positive in this case
    :return:
    '''
    #if working_node == None:
    #    working_node = Node(findBestAttribute(data, attributes))



    if len(data) == 0:
        #working_node.solved = default
        #return working_node
        return default
    if check_homegeneity(data):
        #working_node.solved = data[0][0]
        #return working_node
        return data[0][0]
    if len(attributes) == 0:
        #working_node.solved = find_majority_label(data)
        #return working_node
        return find_majority_label(data)

    best_attr = findBestAttribute(data, attributes)

    tree = {}
    tree['name'] = best_attr
    #print(best_attr)

    for value in attributes[best_attr]:
        new_data = create_subset(data, best_attr, value)
        new_attributes = deepcopy(attributes)
        del new_attributes[best_attr]
        tree[value] = ID3(new_data, new_attributes)
        #working_node.nextNode(child, value)

    return tree


def find_majority_label(data):
    '''
    Finds and returns the majority label
    :param data: data to evaluate
    :return: majority label
    '''
    count_dict = defaultdict(int)
    for d in data:
        count_dict[d[0]] += 1
    return max(count_dict, key=count_dict.get)


def predict(tree, point):
    '''
    Given a tree and a data point, outputs the predicted value fo that data point
    :param tree: tree generated from training
    :param point: 1 data point
    :return: e or p
    '''
    # tree is nested dicts
    # point is one line of data

    attr = tree['name']
    val = tree[point[attr]]
    while type(val) is dict:
        tree = val
        #print(tree)
        attr = tree['name']
        val = tree[point[attr]]
    return val


def train(training_data):
    '''
    takes the training data and returns a decision tree data structure or ADT.
    '''

    attr = initialize_attributes()
    return ID3(training_data, attr)


def classify(tree, data):
    '''
    takes the tree data structure/ADT and labeled/unlabeled data and 
    return a List of classifications.
    '''
    res = []

    for d in data:
        res.append(predict(tree, d))

    return res


def evaluate(actual, predicted):
    '''
    takes a List of actual labels and a List of predicted labels
    and returns the error rate.
    '''
    # actual is all of the test data?
    if len(actual) != len(predicted):
        print("Error mismatch")
        return
    else:
        total = len(actual)
        count = 0
        for i, j in zip(actual, predicted):
            if i[0] != j:
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

def check_homegeneity(data):
    '''
    Checks if the given data is homogenous
    :param data: input data
    :return: Boolean
    '''
    prev = data[0][0]
    for d in data:
        if d[0] != prev:
            return False
    return True

def cross_validate(data):
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
        #actual_results = np.array(test_data)[:,0]

        error_rate = evaluate(test_data, predicted_results)
        all_error_rate.append(error_rate)

        print("The error rate with fold " + str(i + 1) + " as the test set is: " + str(error_rate))

    average_error = sum(all_error_rate) / len(all_error_rate)

    print("The average error rate across all folds is " + str(average_error))

    return


if __name__ == '__main__':
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'

    data = read_data('agaricus-lepiota.data')

    cross_validate(data)
    #print(calc_IG(data, 2))
    #print(find_majority_label(data))
    #printTree(train(data))
    #tree = train(data)
    #print(tree)

    #my_string = "e,x,s,g,f,n,f,w,b,k,t,e,s,s,w,w,p,w,o,e,n,a,g"
    #dp = [x.strip() for x in my_string.split(',')]
    #for d in data:
    #    print(d[0] == predict(tree, d))
    #print(predict(tree, dp))

    #cross_validate(data)

    #attributes = initialize_attributes()
    #print(findBestAttribute(data, attributes))