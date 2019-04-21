import sys
import tokenize
from io import StringIO
from itertools import chain
import collections

def atom(next, token):
    if token[1] == '(':
        out = []
        token = next()
        while token[1] != ')':
            out.append(atom( next, token))
            token = next()
            if token[1] == ' ':
                token = next()
        return out
    elif token[1] == '?':
        token = next()
        return "?" + token[1]
    else:
        return token[1]

def parse(exp):
    src = StringIO(exp).readline
    tokens = tokenize.generate_tokens(src)
    return atom(tokens.__next__, tokens.__next__())

'''
Takes in a parsed, potentially nested list or string and reduces it to logic
:param {List or String} exp: input list or string to be converted
:return {String}: converted string
'''
def unparse(exp):
    print(exp)
    if isinstance(exp, list):
        exp = flatten(exp)
        return '('  + ' '.join(exp) + ')'
    return exp

'''
Flattens an arbitrarily nested list
:param {List} items: input list to be flattened
:return {List}: flattened list
'''
# Flattening lists from https://stackoverflow.com/questions/10823877/what-is-the-fastest-way-to-flatten-arbitrarily-nested-lists-in-python
def flatten(items, seqtypes=(list, tuple)):
    for i, x in enumerate(items):
        while i < len(items) and isinstance(items[i], seqtypes):
            items[i:i+1] = items[i]
    return items

def assert_parse(src, trg, debug):
    result = parse(src)
    if debug:
        print(src, result, trg)
    assert result == trg 


def test_parser(debug):
    assert_parse('Fred', 'Fred', debug)
    assert_parse('?x', '?x', debug)
    assert_parse("(loves Fred ?x)", ['loves', 'Fred', '?x'], debug)
    assert_parse("(father_of Barney (son_of Barney))", ['father_of', 'Barney', ['son_of', 'Barney']], debug)


def is_variable(exp):
    return isinstance(exp, str) and exp[0] == "?"


def is_constant(exp):
    return isinstance(exp, str) and not is_variable(exp)

'''
Unification function implemented using pseudocide in assignment pdf
Recursively unifies two expressions represented as nested lists
:param {List} list_expression1: 1st expression that has been parsted into a list
:param {List} list_expression2: 2nd expression that has been parsted into a list
:return {Dict}: dictionary of key = variable and value = variable assignment
'''
def unification(list_expression1, list_expression2):
    ### YOUR SOLUTION HERE ###
    # implement the pseudocode in 2.3 of the assignment PDF
    ### YOUR SOLUTION HERE ### 

    if (is_constant(list_expression1) or not list_expression1) and (is_constant(list_expression2) or not list_expression2):
        if list_expression1 == list_expression2:
            return {}
        else:
            return None
    if is_variable(list_expression1):
        if list_expression1 in list_expression2:
            return None
        else:
                return {str(list_expression1): unparse(list_expression2) }
    if is_variable(list_expression2):
        if list_expression2 in list_expression1:
            return None
        else:
            return {str(list_expression2): unparse(list_expression1) }

    first1 = list_expression1[0]
    first2 = list_expression2[0]
    result1 = unification(first1, first2)
    if result1 == None:
        return None
    result2 = unification(list_expression1[1:], list_expression2[1:])
    if result2 == None:
        return None


    #res = [result1, flatten(result2, [])]
    #f = flatten(res, [])
    return merge_dict(result1, result2)

'''
Helper function that merges two dicts into one
Also checks for failure in merging and simplifies expressions via variable substitution
:param {Dict} d1: 1st dictionary to be merged
:param {Dict} d2: 2nd dictionary to be merged
:return {Dict}: merged dictionary
'''
def merge_dict(d1, d2):
    # checks to see if None was assigned
    if d1 == None or d2 == None:
        return None

    d = {}
    # merging dictionary
    for k1 in d1:
        d[k1] = d1[k1]
    for k2 in d2:
        if k2 in d:
            if d1[k2] != d2[k2]:
                return None
        else:
            d[k2] = d2[k2]

    # replaces variables in constants
    for key in d:
        for key2, values in d.items():
            if key in values:
                d[key2] = values.replace(key, d[key])
    return d

def test(d):
    for key in d:
        for key2, values in d.items():
            if key in values:
                d[key2] = values.replace(key, d[key])
    return d

def unify(s_expression1, s_expression2):
    return unification(parse(s_expression1), parse(s_expression2))

unifications = 0
def assert_unify(exp1, exp2, trg, debug):
    # never do this for real code!
    global unifications
    result = unify(exp1, exp2)
    if debug:
        print(unifications, exp1, exp2, result, trg)
    assert result == trg 
    unifications += 1

def test_unify(debug):
    
    assert_unify('Fred', 'Barney', None, debug)
    # use underscores instead of hypthens
    assert_unify('(quarry_worker Fred)', '(quarry_worker ?x)', {"?x": 'Fred'}, debug)
    # add the remainder of the self check
    #
    # add 5 additional test cases.
    #
    # Self Check Problems
    assert_unify('Pebbles', 'Pebbles', {}, debug)
    assert_unify('(son Barney ?x)', '(son ?y Bam_Bam)', {"?y": 'Barney', "?x": 'Bam_Bam'}, debug)
    assert_unify('(married ?x ?y)', '(married Barney Wilma)', {"?x": 'Barney', "?y": 'Wilma'}, debug)
    assert_unify('(son Barney ?x)', '(son ?y (son Barney))', {"?y": 'Barney', "?x": '(son Barney)'}, debug)
    assert_unify('(son Barney ?x)', '(son ?y (son ?y))', {"?y": 'Barney', "?x": '(son Barney)'}, debug)
    assert_unify('(son Barney Bam-Bam)', '(son ?y (son Barney))', None, debug)
    assert_unify('(loves Fred Fred)', '(loves ?x ?x)', {"?x": 'Fred'}, debug)
    assert_unify('(future George Fred)', '(future ?y ?y)', None, debug)

    #Additional test cases
    assert_unify('(future George Fred)', '(son ?x ?y)', None, debug)
    assert_unify('(son Barney ?y)', '(son ?y (son ?y))', None, debug)
    assert_unify('(son ?y (son ?y))', '(son ?y (son ?y))', None, debug)
    assert_unify('(son Barney (son Barney))', '(son Barney (son Barney))', {}, debug)
    assert_unify('(son Barney (son (son (son ?y))))', '(son ?y ?x)', {"?y": 'Barney', "?x": '(son son son Barney)'}, debug)

if __name__ == "__main__":
    
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'

    test_parser(debug)
    test_unify(debug)
    print(unifications) # should be 15
    

