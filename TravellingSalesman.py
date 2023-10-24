import random
import pandas as pd
import numpy as np

class TravellingSalesmanProblem:
    def __init__(self):
        dm = pd.read_csv('tour50(1).csv', sep=',', header=None)
        self.dm = dm.to_numpy()

class Salesman:
    def __init__(self, tsp):
        cities = len(tsp.dm)  # amount of cities
        self.path = random.sample(range(1, cities+1), cities)  # generates a random array of the cities
        # calculate distance of path
        d_array = [tsp.dm[self.path[i]-1, self.path[i+1]-1] for i in range(0, len(self.path)-1)] + [tsp.dm[self.path[-1]-1, self.path[1]-1]]
        self.d = sum(d_array)


tsp = TravellingSalesmanProblem()
index = 0
while index < 1000:
    s = Salesman(tsp)
    if s.d != np.inf:
        print(s.d)
    index += 1
