import random as rd
import pandas as pd
import numpy as np

''' creates a random TSP problem '''
class TravellingSalesmanProblem:
    def __init__(self):
        dm = pd.read_csv('tour50(1).csv', sep=',', header=None)
        self.dm = dm.to_numpy()

''' creates an individual '''
class Salesman:
    def __init__(self,tsp,path=None):
        # for generating first population
        if path is None:
            cities = len(tsp.dm)                                # amount of cities
            self.path = rd.sample(range(1, cities+1), cities)   # generates a random array of the cities

        # for generating crossover children
        else:
            self.path = path

        # calculate distance of path
        d_array = [tsp.dm[self.path[i]-1, self.path[i+1]-1] for i in range(0, len(self.path)-1)] + [tsp.dm[self.path[-1]-1, self.path[1]-1]]
        self.d = sum(d_array)

''' returns a value between 0 and 1 '''
def fitness(sm):
    fit = 50000/sm.d
    if fit > 1: fit = 1; print('overflow')
    return fit

''' reverses a portion of the path with chance alpha '''
def mutate(sm,alpha):
    if rd.random() < alpha:
        end = rd.randint(0, len(sm.path) - 1)
        size = rd.randint(2, len(sm.path))
        sel_cities = [sm.path[i] for i in range(end, end - size, -1)]
        for i in range(len(sel_cities)):
            sm.path[end - size + 1 + i] = sel_cities[i]

''' crossover using algorithm of book chapter 5.3 '''
def crossover(sm1,sm2,tsp):
    new_path = [None] * len(sm1.path)
    end = rd.randint(0, len(sm1.path) - 1)
    size = rd.randint(2, len(sm1.path))

    # step 1 (see chapter 5.3)
    for i in range(end, end-size, -1):
        new_path[i] = sm1.path[i]

    # step 2
    for i in range(end,end-size,-1):
        if not sm2.path[i] in new_path:
            crossover_operation(new_path,sm1.path,sm2.path,i,sm2.path[i])

    # step 3
    for i in range(len(new_path)):
        if new_path[i] is None and not sm2.path[i] in new_path:
            new_path[i] = sm2.path[i]

    # create new salesman
    return Salesman(tsp, new_path)

''' implements step 2 of crossover '''
def crossover_operation(newp,p1,p2,i,city):
    while True:
        index = p2.index(p1[i])
        if newp[index] is None:
            newp[index] = city
            return
        else:
            return crossover_operation(newp,p1,p2,index,city)

''' k-tournament selection (no elitism)'''
def selection(pop,k):
    selected = rd.sample(pop, k)  # select k individual without replacement
    ind = np.argmax(list(map(lambda x: fitness(x), selected)))  # find index of individual with highest fitness
    return selected[ind]

''' combines the population with the offspring and returns the l best individuals'''
def elimination(pop,offspring,lam):
    pop.extend(offspring)               # concatenates population and offspring
    pop.sort(key=fitness,reverse=True)  # sorts on fitness, descending in value
    return pop[:lam]
