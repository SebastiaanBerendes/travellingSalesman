import random
import numpy as np 
import time
import Reporter
import math
import pandas as pd


# Modify the class name to match your student number.
class r0753150:
    def __init__(self, k, mu, l, a, var):
        self.Ksample = k
        self.populationNb = mu
        self.childrenNb = l
        self.alpha = a
        self.variety = var
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    def optimize(self, filename):
        file = open(filename)
        self.distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()
        self.length = len(self.distanceMatrix)

        self.population, self.distance = self.initParent()
        
        # Your code here.
        yourConvergenceTestsHere = True
        self.iterationNb = 0
        same = 0
        prev_best = np.inf

        for _ in range(2000):
            offsprings, fit = self.generateOffsprings()
            self.population = np.vstack((self.population,offsprings))
            self.distance = np.concatenate((self.distance,fit))
            self.mutate()
            self.eliminate()

            meanObjective = sum(1/self.distance)/self.populationNb
            bestObjective = min(self.distance)
            if bestObjective < prev_best:
                same = 0
                prev_best = bestObjective
            elif bestObjective == prev_best:
                same+= 1
            bestSolution = self.population[np.where(self.distance==min(self.distance))[0][0]]

            print("Iteration: " + str(self.iterationNb) + ", best fitness: " + str(
                bestObjective) + ", mean fitness: " + str(meanObjective))
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            self.iterationNb += 1
            if same == 200:
                break
            if timeLeft < 0:
                break
        return bestObjective
    def initParent(self):
        population = np.empty((self.populationNb, self.length), dtype=np.int64)
        fit = np.empty((self.populationNb), dtype=np.float64)
        for j in range(self.populationNb): #loop over the amount of parents 
            distance = 0
            cities = list(range(self.length))
            random.shuffle(cities)
            city = random.choice(cities)
            tour = np.zeros(len(cities), dtype=np.int64)
            tour[0] = city
            for i in range(self.length-1):
                cities.remove(city)
                for nextCity in cities:
                    if self.distanceMatrix[city][nextCity] < np.inf:
                        break
                tour[i+1] = nextCity
                distance += self.distanceMatrix[city][nextCity]
                city = nextCity
            population[j] = tour
            fit[j] = distance
        return population, fit
    
    def fitness(self, tour):
        totalDist = 0
        for i in range(self.length-1):
            current = tour[i]
            next = tour[(i+1)% (len(tour))]
            dist = self.distanceMatrix[current][next]
            totalDist += dist
        return totalDist
    
    def parentSelection(self):
        indices = random.sample(range(self.populationNb), self.Ksample)
        best = np.where(self.distance==(min(self.distance[indices])))[0][0]
        return self.population[best]
    
    def mergeParents(self, p1, p2):
        size = random.randint(0, self.length-1)
        endCity = random.randint(0, self.length-1)
        child = [None]*self.length
        for i in range(endCity-size, endCity):
            child[i] = p1[i]

        for i in range(endCity-size, endCity):
            if not p2[i] in child:
                self.crossover_operation(child, p1,p2,i,p2[i])

        for i in range(self.length):
            if child[i] is  None:
                child[i] = p2[i]
        return np.array(child)

    def crossover_operation(self, child,p1,p2,i, city):
        while True:
            index = np.where(p2==p1[i])[0][0]
            if child[index] is None:
                child[index] = city
                return
            else:
                return self.crossover_operation(child, p1,p2, index, city)
            

    def generateOffsprings(self):
        offsprings = np.empty((self.childrenNb,self.length),dtype=np.int64)
        fit = np.empty((self.childrenNb), dtype=np.float64)
        t1 =time.time()
        for i in range(self.childrenNb):
            p1 = self.parentSelection()
            p2 = self.parentSelection()
            child = self.mergeParents(p1, p2)
            dist = self.fitness(child)
            fit[i] = dist
            offsprings[i] = child
        return offsprings, fit   
    
    def mutate(self):
        for ind, individual in enumerate(self.population):
            if random.random() < self.alpha:
                size = random.randint(0, len(individual) - 1)
                endCity = random.randint(0, len(individual) - 1)
                reversedCity = [individual[i] for i in range(endCity, endCity - size, -1)]
                for i in range (len(reversedCity)):
                    individual[endCity - size +1 +i] = reversedCity[i]
                self.distance[ind] = self.fitness(individual)

    def eliminate(self):
        index = np.argsort(self.distance)
        population = self.population[index]
        distance = self.distance[index]
        self.population = population[:int(self.variety*self.populationNb)]
        self.distance = distance[:int(self.variety*self.populationNb)]
        while len(self.population) < self.populationNb:
            ind = random.randint(int(self.variety*self.populationNb), self.populationNb+self.childrenNb-1)
            self.population = np.concatenate((self.population, np.expand_dims(population[ind], axis=0)))
            self.distance = np.concatenate((self.distance, np.expand_dims(distance[ind], axis=0)))


if __name__ == '__main__':
    results = {}
    a = 0.05
    tournament = [3]
    parents = [100,200,500,1000]
    children = [100,200,500,1000]
    variety = [00.75,0.9]
    dfColumns = ["k", "mu_size", "lambda_size", "mutation_rate", "variety","mean_distance", "min_distance", "mean_runtime"]
    data = pd.DataFrame(columns = dfColumns)
    for tour in tournament:
        for mu in parents:
            for l in children:
                for var in variety:
                    result = []
                    tijd = []
                    for _ in range(10):
                        t0 = time.time()
                        res = r0753150(tour,mu,l,a, var).optimize('tour50.csv')
                        t1 = time.time()
                        result.append(res)
                        tijd.append(t1-t0)
                    par = [tour,mu,l,a,var, sum(result)/len(result), min(result), sum(tijd)/len(tijd)]
                    data.loc[len(data)] = par
    data.to_csv("data_TSM.csv", index=False)