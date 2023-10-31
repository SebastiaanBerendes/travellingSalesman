import random as rd
import pandas as pd
import numpy as np
import time

file = 100  # parameter for file

''' creates a random TSP problem '''
class TravellingSalesmanProblem:
    def __init__(self):
        dm = pd.read_csv('tour'+str(file)+'.csv', sep=',', header=None)
        self.dm = dm.to_numpy()

''' creates an individual '''
class Salesman:
    def __init__(self,tsp,path=None):
        # for generating first population
        if path is None:
            inf = False
            self.d = 0
            n = len(tsp.dm)
            cities = list(range(1,n+1))
            city = (rd.choice(cities))
            self.path = [city]
            cities.pop(city-1)
            for counter in range(n-1):
                city = (rd.choice(cities))
                counter = 0
                while tsp.dm[self.path[-1]-1,city-1] == np.inf and counter < (1/2)*n and not inf:
                    city = (rd.choice(cities))
                    counter += 1
                if counter >= (1/2)*n:
                    inf = True
                self.d += tsp.dm[self.path[-1]-1,city-1]
                self.path.append(city)
                cities.remove(city)
            self.d += tsp.dm[self.path[-1]-1,self.path[1]-1]

        # for generating crossover children
        else:
            self.path = path
            # calculate distance of path
            d_array = [tsp.dm[self.path[i]-1, self.path[i+1]-1] for i in range(0, len(self.path)-1)] + [tsp.dm[self.path[-1]-1, self.path[1]-1]]
            self.d = sum(d_array)

''' returns a value between 0 and 1 '''
def fitness(sm):
    val = np.inf
    match file:
        case 50: val = 15000
        case 100: val = 45000
        case 200: val = 20000
        case 500: val = 75000
        case 750: val = 100000
        case 1000: val = 100000
    fit = val/sm.d
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

''' Initializes the population '''
def initialize(tsp, lam):
    return [Salesman(tsp) for i in range(lam)]

def termination(meanlist):
    return len(set(meanlist[-50:])) == 1

def evolutionary_algorithm(tsp):
    start = time.time()
    lam = 1000; mu = 1000; its = 1000; alpha = 0.05; k = 3

    meanlist = []
    population = initialize(tsp, lam)
    for i in range(its):
        # Recombination
        offspring = []
        for j in range(mu):
            p1 = selection(population,k)
            p2 = selection(population,k)
            offspring.append(crossover(p1,p2,tsp))
            mutate(offspring[-1],alpha)

        # Mutation
        for ind in population:
            mutate(ind,alpha)

        # Elimination
        population = elimination(population,offspring,lam)

        # Prints
        fitnessess = list(map(lambda x: fitness(x),population))
        index = np.argmax(fitnessess)
        mean = np.mean(fitnessess)
        print("Iteration",i)
        print("Mean fitness:",mean.round(4),"and Max fitness:",str(max(fitnessess).round(4))+'/'+str(int(population[index].d)))

        # Termination check
        meanlist.append(mean)
        if i > 50:
            stop = termination(meanlist)
            if stop:
                break
    print('-------------------------------------')
    print('Best invdividual after '+str(i)+' iterations')
    print(population[np.argmax(list(map(lambda x: fitness(x),population)))].path)
    print(population[np.argmax(list(map(lambda x: fitness(x),population)))].d)
    end = time.time()
    print('Total elapsed time:',str(round(end-start,4)),'seconds')

TSP = TravellingSalesmanProblem()
evolutionary_algorithm(TSP)
