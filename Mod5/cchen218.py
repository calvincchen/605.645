import sys
from random import gauss, random, getrandbits, sample, randint, uniform
from copy import deepcopy
#from IPython.core.display import *
#from StringIO import StringIO

# There is a way to do this assignment with higher order functions
# so that you only need one function to do the ga() and you delegate
# the brains of `binary_ga` and `real_ga` to it. 
#
# If this means nothing to you, ignore it. :D

'''
Randomly generate a number of binary individual genotypes to make initial population
@param {int} population_size: number of individuals to generate
@return {List{String}}: All individuals in the population
'''
def generate_population_bin(population_size):
    population = []
    # creating population of individuals of 100 bits
    # population is a 100 element array
    for _ in range(population_size):
        population.append("{0:b}".format(getrandbits(100)))
    return population

'''
Randomly generate a number of real individual genotypes to make initial population
@param {int} population_size: number of individuals to generate
@return {List{List{Float}}}: All individuals in the population
'''
def generate_population_real(population_size):
    population = []

    for _ in range(population_size):
        individual = []
        for _ in range(10):
            individual.append(uniform(-5.12, 5.12))
        population.append(individual)
    return population


'''
Tournament selection for picking parents from a population
@param {List{String}} population: All individuals in the population
@return {String} parent1, parent2: Best parents
'''
def pick_parents_tournament_bin(population):
    n = 14
    random_parents = sample(population, n)
    # sort with evaluate_bin as key, ordered from largest to smallest
    random_parents.sort(key = lambda x: -evaluate_bin(x, True))


    parent1 = random_parents[0]
    parent2 = random_parents[1]
    return parent1, parent2

'''
Tournament selection for picking parents from a population
@param {List{List{Float}}} population: All individuals in the population
@return {List{Float}} parent1, parent2: Best parents
'''
def pick_parents_tournament_real(population):
    n = 14
    random_parents = sample(population, n)
    # sort with evaluate_bin as key, ordered from largest to smallest
    random_parents.sort(key = lambda x: -evaluate_real(x, True))

    parent1 = random_parents[0]
    parent2 = random_parents[1]
    return parent1, parent2

'''
Reproductive function for binary GA
@param {String} parent1: Parent1's genotype
@param {String} parent2: Parent2's genotype
@param {Float} Pc: Predefined probability cutoff for crossover
@param {Float} Pm: Predefined probability cutoff for mutation
@return {String} son, daughter: Children of the parents
'''
def reproduce_bin(parent1, parent2, Pc, Pm):
    # crossover
    son, daughter = crossover_bin(parent1, parent2, Pc)
    # mutation
    son = mutation_bin(son, Pm)
    daughter = mutation_bin(daughter, Pm)
    return son, daughter

'''
Reproductive function for real GA
@param {List{Float}} parent1: Parent1's genotype
@param {List{Float}} parent2: Parent2's genotype
@param {Float} Pc: Predefined probability cutoff for crossover
@param {Float} Pm: Predefined probability cutoff for mutation
@return {List{Float}} son, daughter: Children of the parents
'''
def reproduce_real(parent1, parent2, Pc, Pm):
    # crossover
    son, daughter = crossover_real(parent1, parent2, Pc)
    # mutation
    son = mutation_real(son, Pm)
    daughter = mutation_real(daughter, Pm)
    return son, daughter

'''
Function to evaluate the crossover between binary parents
@param {String} parent1: Parent1's genotype
@param {String} parent2: Parent2's genotype
@param {Float} probability: Predefined probability cutoff for crossover
@return {String} son, daughter: Children of the parents
'''
def crossover_bin(parent1, parent2, probability):
    crossover_value = random()
    crossover_pos = randint(0, 99)
    if probability > crossover_value: # and crossover_pos != 0 and crossover_position != 99:
        son = parent1[:crossover_pos] + parent2[crossover_pos:]
        daughter = parent2[:crossover_pos] + parent1[crossover_pos:]
    else:
        son = parent1
        daughter = parent2
    return son, daughter

'''
Function to evaluate the crossover between real parents
@param {List{Float}} parent1: Parent1's genotype
@param {List{Float}} parent2: Parent2's genotype
@param {Float} probability: Predefined probability cutoff for crossover
@return {List{Float}} son, daughter: Children of the parents
'''
def crossover_real(parent1, parent2, probability):
    crossover_value = random()
    crossover_pos = randint(0, 10)
    if probability > crossover_value:
        son = parent1[:crossover_pos] + parent2[crossover_pos:]
        daughter = parent2[:crossover_pos] + parent1[crossover_pos:]
    else:
        son = parent1
        daughter = parent2
    return son, daughter

'''
Function to evaluate possible mutation of an individual;
@param individual {String}: String representation of 100-bit genotype of an individual'
@param probability {Float}: Predefined probability cutoff for mutations
@return {String}: (Un)modified individual's genotype, depending on random chance
'''
def mutation_bin(individual, probability):
    mutation_value = random()
    mutation_pos = randint(0, 100)

    if probability > mutation_value:
        individual = individual[:mutation_pos] + str(randint(0, 1)) + individual[mutation_pos + 1:]
    return individual

'''
Function to evaluate possible mutation of an individual;
@param individual {List{Float}}: An individual's genotype
@param probability {Float}: Predefined probability cutoff for mutations
@return {List{Float}}: (Un)modified individual's genotype, depending on random chance
'''
def mutation_real(individual, probability):
    mutation_value = random()
    mutation_pos = randint(0, 9)

    if probability > mutation_value:
        individual[mutation_pos] = individual[mutation_pos] * gauss(0,1)
    return individual

'''
Function to help GA evaluate for smallest result
@param {Float} value: input value
@return {Float}: minimized value
'''
def minimizer(value):
    return 1 / (1 + value)

'''
Transforms a 100-bit genotype into a float phenotype per project description
@param {String} genotype: individual's genotype as 100-bit binary string
@return {Float}: Phenotype within range (-5.12, 5.12)
'''
def get_phenotype(genotype):
    return int(genotype, 2) - 512 / 100

'''
Evaluates the value of a Binary individual genotype
@param {String} individual: individual to be evaluated
@param {Boolean} minimize: True if we want to return minimized value, False if return shifted value
@return {Float}: minimized or shifted value
'''
def evaluate_bin(individual, minimize = True):
    # Converts 100 bit binary string to list of 10 10-bit numbers
    spherical_dimensions = [get_phenotype(individual[i:i+10]) for i in range(0, len(individual), 10)]

    shifted_value = shifted_sphere(0.5, spherical_dimensions)

    if minimize:
        return minimizer(shifted_value)
    else:
        return shifted_value

'''
Evaluates the value of a Float individual genotype
@param {List{Float}} individual: individual to be evaluated
@param {Boolean} minimize: True if we want to return minimized value, False if return shifted value
@return {Float}: minimized or shifted value
'''
def evaluate_real(individual, minimize = True):
    shifted_value = shifted_sphere(0.5, individual)

    if minimize:
        return minimizer(shifted_value)
    else:
        return shifted_value

'''
Evaluates the best individual in a population of 100-bit binary strings
@param {List} population: List of individuals in current population
@return {String}: Individual with largest evaluated value
''' 
def best_individual_bin(population):
    sorted_pop = deepcopy(population)
    sorted_pop.sort(key = lambda x: -evaluate_bin(x))
    return sorted_pop[0]

'''
Evaluates the best individual in a population of floats
@param {List} population: List of individuals in current population
@return {Float}: Individual with largest evaluated value
'''
def best_individual_real(population):
    sorted_pop = deepcopy(population)
    sorted_pop.sort(key = lambda x: -evaluate_real(x))
    return sorted_pop[0]

'''
Print statements required for debugging
@param {List} population: List of individuals in current population
@param {Int} generation: Current generation number
@return None
'''
def debug_bin(population, generation):
    print("Generation Number: ", generation)
    best_individual = best_individual_bin(population)
    print("Best Individual Genotype: ", best_individual)
    best_phenotype = [get_phenotype(best_individual[i:i+10]) for i in range(0, len(best_individual), 10)]
    print("Best Individual Phenotype: ", best_phenotype)
    print("Fitness of Best Individual: ", evaluate_bin(best_individual))
    print("Function Value (Shifted Sphere Function) of Best Individual: ", evaluate_bin(best_individual, False))
    print()
    return

'''
Print statements required for debugging
@param {List} population: List of individuals in current population
@param {Int} generation: Current generation number
@return None
'''
def debug_real(population, generation):
    print("Generation Number: ", generation)
    best_individual = best_individual_real(population)
    print("Best Individual Genotype: ", best_individual)
    print("Best Individual Phenotype: ", best_individual)
    print("Fitness of Best Individual: ", evaluate_real(best_individual))
    print("Function Value (Shifted Sphere Function) of Best Individual: ", evaluate_real(best_individual, False))
    print()
    return

'''
Runs the Binary GA Algorithm
@param {Dictionary} parameters: Input parameters
@param {Boolean} debug: Boolean to see if generational data should be printed
@return {Float}: Best individual in the final generation
'''
def binary_ga(parameters, debug = False):
    population = generate_population_bin(parameters['population_size'])
    generations = 0
    crossover_rate = parameters['crossover_rate']
    mutation_rate = parameters['mutation_rate']

    while generations < parameters['generations']:
        if debug:
            debug_bin(population, generations)
        next_population = []
        for _ in range(int(len(population)/2)):
            parent1, parent2 = pick_parents_tournament_bin(population)
            child1, child2 = reproduce_bin(parent1, parent2, crossover_rate, mutation_rate)
            next_population.append(child1)
            next_population.append(child2)
        population = next_population
        generations +=1
    debug_bin(population, generations)
    return best_individual_bin(population)
    
'''
Runs the Real GA Algorithm
@param {Dictionary} parameters: Input parameters
@param {Boolean} debug: Boolean to see if generational data should be printed
@return {Float}: Best individual in the final generation
'''
def real_ga(parameters, debug = False):
    population = generate_population_real(parameters['population_size'])
    generations = 0
    crossover_rate = parameters['crossover_rate']
    mutation_rate = parameters['mutation_rate']

    while generations < parameters['generations']:
        if debug:
            debug_real(population, generations)
        next_population = []
        for _ in range(int(len(population)/2)):
            parent1, parent2 = pick_parents_tournament_real(population)
            child1, child2 = reproduce_real(parent1, parent2, crossover_rate, mutation_rate)
            next_population.append(child1)
            next_population.append(child2)
        population = next_population
        generations +=1
    debug_real(population, generations)
    return best_individual_real(population)

def shifted_sphere( shift, xs):
    return sum( [(x - shift)**2 for x in xs])

if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'

    parameters = {
        "f": lambda xs: shifted_sphere( 0.5, xs),
        "minimization": True,
        "population_size": 100, 
        "generations": 5000,
        "mutation_rate": .05, 
        "crossover_rate": .9
        # put other parameters in here, add , to previous line.
    }
    print("Executing Binary GA")
    binary_ga(parameters, debug)

    print("Executing Real-Valued GA")
    parameters = {
        "f": lambda xs: shifted_sphere( 0.5, xs),
        "minimization": True,
        "population_size": 100, 
        "generations": 200,
        "mutation_rate": .05, 
        "crossover_rate": .9
        # put other parameters in here, add , to previous line.
    }
    real_ga(parameters, debug)

