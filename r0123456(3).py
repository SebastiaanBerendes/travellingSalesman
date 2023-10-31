import random
import time 
import Reporter
import numpy as np

# Modify the class name to match your student number.
class r0123456:

	def __init__(self):
		self.Nbparents = 2
		self.KSample = 4
		self.parents = 100
		self.bestNB = 75
		self.offsprings = 200
		self.alpha = 0.05
		self.reporter = Reporter.Reporter(self.__class__.__name__)

	# The evolutionary algorithm's main loop
	def optimize(self, filename):
		# Read distance matrix from file.		
		file = open(filename)
		self.distanceMatrix = np.loadtxt(file, delimiter=",")
		file.close()

		parents = self.initParent()

		# Your code here.
		yourConvergenceTestsHere = True
		self.iterationNb = 0
		while( yourConvergenceTestsHere ):
			meanObjective = self.meanFitness(parents)
			bestObjective = self.bestFitness(parents)
			bestSolution = self.bestTour(parents)
			
			offsprings = self.generateOffsprings(parents)
			joinedPopulation = parents + offsprings
			population = self.mutate(joinedPopulation)
			parents = self.eliminate(population)
			print("Iteration: " + str(self.iterationNb) + ", best fitness: " + str(
                bestObjective) + ", mean fitness: " + str(meanObjective))
			# Your code here.

			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
			timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			self.iterationNb += 1
			if timeLeft < 0:
				break

		# Your code here.
		return 0
	
	def meanFitness(self, parents):
		fitness = [self.fitness(tour) for tour in parents]
		return sum(fitness) / len(fitness)

	def bestFitness(self, parents):
		fitness = [self.fitness(tour) for tour in parents]
		return min(fitness)

	def bestTour(self, parents):
		fitness = [self.fitness(tour) for tour in parents]
		return parents[fitness.index(min(fitness))]

	def fitness(self, tour):
		totalDist = 0
		for i in range(len(tour)-1):
			initCity = tour[i]
			nextCity = tour[(i+1)% (len(tour))]
			dist = self.distanceMatrix[initCity][nextCity]
			if dist == np.inf:
				# totalDist = np.inf 
				totalDist = 1000000000 # put very large number otherwise mean distance also inf if np.inf
				break
			totalDist += dist
		return totalDist
	
	def initParent(self):
		parent = []
		for _ in range(self.parents): #loop over the amount of parents 
			cities = list(range(len(self.distanceMatrix)))
			random.shuffle(cities)
			city = random.choice(cities)
			tour = np.zeros(len(cities), dtype=np.int32)
			tour[0] = city
			for i in range(len(self.distanceMatrix)-1):
				cities.remove(city)
				for nextCity in cities:
					if self.distanceMatrix[city][nextCity] < np.inf:
						break
				tour[i+1] = nextCity
				city = nextCity
			parent.append(tour)
		return parent
	
	def parentSelection(self, population):
		Kparents = random.sample(population, self.KSample)
		return self.bestTour(Kparents)
	
	def mergeParents(self, p1, p2):
		size = random.randint(0, len(p1) - 1)
		endCity = random.randint(0, len(p2) - 1)
		child = [None for _ in range(len(self.distanceMatrix))]
		segP2 = []
		segP1 = []
		for i in range(endCity, endCity-size, -1):
			child[i] = p1[i]
			segP2.append(p2[i])
			segP1.append(p1[i])

		for city in segP2: #look for improvement
			if city not in child:
				for i, placement in enumerate(segP1):
					if placement not in segP2:
						index = np.where(p2 == placement)[0][0]
						child[index] = city
						segP1 = np.delete(segP1, i)
						break
		
		for i in range(len(self.distanceMatrix)):
			if child[i] is None:
				child[i] = p2[i]

		return np.array(child)
	
	def generateOffsprings(self, parents):
		offsprings = []
		for _ in range(self.offsprings):
			parent = []
			for _ in range(self.Nbparents):
				parent.append(self.parentSelection(parents))
			offsprings.append(self.mergeParents(parent[0], parent[1]))
		return offsprings

	def eliminate(self, joinedPopulation): #maybe use kfold split in groups of size k and select the min value for every gro
		population = []
		joinedPopulation.sort(key=self.fitness)
		population = joinedPopulation[:self.bestNB]
		while len(population) < self.parents:
			population.append( random.sample(joinedPopulation[self.bestNB:], 1)[0])
		return population


	def mutate(self, population):
		for individual in population:
			if random.random() < self.alpha:
				size = random.randint(0, len(individual) - 1)
				endCity = random.randint(0, len(individual) - 1)
				reversedCity = [individual[i] for i in range(endCity, endCity - size, -1)]
				for i in range (len(reversedCity)):
					individual[endCity - size +1 +i] = reversedCity[i]
		return population

if __name__ == '__main__':
	r0123456().optimize('tour50(1).csv')
