import random as rd
import pandas as pd
import numpy as np

class TravellingSalesmanProblem:
    def __init__(self):
        dm = pd.read_csv('tour50(1).csv', sep=',', header=None)
        self.dm = dm.to_numpy()

class Salesman:
    def __init__(self, tsp):
        cities = len(tsp.dm)  # amount of cities
        self.path = rd.sample(range(1, cities+1), cities)  # generates a random array of the cities
        # calculate distance of path
        d_array = [tsp.dm[self.path[i]-1, self.path[i+1]-1] for i in range(0, len(self.path)-1)] + [tsp.dm[self.path[-1]-1, self.path[1]-1]]
        self.d = sum(d_array)

def mutate(sm, alpha):
    if rd.random() < alpha:
        end = rd.randint(0, len(sm.path) - 1)
        size = rd.randint(2, len(sm.path))
        sel_cities = [sm.path[i] for i in range(end, end - size, -1)]
        for i in range(len(sel_cities)):
            sm.path[end - size + 1 + i] = sel_cities[i]




tsp = TravellingSalesmanProblem()
sm = Salesman(tsp)
print(sm.path)
mutate(sm, 1)
print(sm.path)
