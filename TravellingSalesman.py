import random as rd
import pandas as pd
import numpy as np

class TravellingSalesmanProblem:
    def __init__(self):
        dm = pd.read_csv('tour50(1).csv', sep=',', header=None)
        self.dm = dm.to_numpy()

class Salesman:
    def __init__(self, tsp, path=None):  # construtctor for first population
        if path is None:
            cities = len(tsp.dm)  # amount of cities
            self.path = rd.sample(range(1, cities+1), cities)  # generates a random array of the cities
        else:
            self.path = path
        # calculate distance of path
        d_array = [tsp.dm[self.path[i]-1, self.path[i+1]-1] for i in range(0, len(self.path)-1)] + [tsp.dm[self.path[-1]-1, self.path[1]-1]]
        self.d = sum(d_array)

def fitness(sm):
    return 50000/sm.d

def mutate(sm, alpha):
    if rd.random() < alpha:
        end = rd.randint(0, len(sm.path) - 1)
        size = rd.randint(2, len(sm.path))
        sel_cities = [sm.path[i] for i in range(end, end - size, -1)]
        for i in range(len(sel_cities)):
            sm.path[end - size + 1 + i] = sel_cities[i]

def crossover(sm1, sm2, tsp):
    new_path = [None] * len(sm1.path)
    end = rd.randint(0, len(sm1.path) - 1)
    size = rd.randint(2, len(sm1.path))
    # step 1
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

def crossover_operation(np,p1,p2,i,city):
    while True:
        index = p2.index(p1[i])
        if np[index] is None:
            np[index] = city
            return
        else:
            return crossover_operation(np,p1,p2,index,city)

def selection(tsp,population, k):
    selected = rd.sample(population, k)
    ind = np.argmax(list(map(lambda x: fitness(x), selected)))
    return selected[ind]


tsp = TravellingSalesmanProblem()
pop = []
k = 1000
for i in range(k):
    sm = Salesman(tsp)
    pop.append(sm)
selection(tsp, pop, k)
