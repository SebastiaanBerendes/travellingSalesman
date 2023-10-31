# %%
import numpy as np

# %%
def load_distance_matrix(data):
    distance_matrix = np.genfromtxt(data, delimiter=',')
    
    return distance_matrix

# %%
distance_matrix = load_distance_matrix('tour50(1).csv')
distance_matrix.shape

# %%
def route(permutation: list, mutation_rate: int = 0.01) -> dict:

    route = dict()

    route["order"] = permutation
    route["mutation_rate"] = mutation_rate

    return route

# %%
def calculate_total_distance(distance_matrix: np.ndarray, route: list):

    n = len(route)
    total_distance = 0
    
    for i in range(n - 1):
        current_node = route[i]
        next_node = route[i + 1]
        distance = distance_matrix[current_node, next_node]
        
        if distance == np.inf:
            return np.inf
        else:
            total_distance += distance_matrix[current_node, next_node]

    total_distance += distance_matrix[route[-1], route[0]]

    return total_distance

# %%
def generate_valid_route(distance_matrix) -> np.array:

    n = distance_matrix.shape[0]
    routes = []

    while len(routes) < 1:

        route = []
        temp_route = list(range(n))
        parking = []
        current_node = np.random.choice(n)
        route.append(current_node)
        temp_route.remove(current_node)

        while len(route) < n:
            next_node = np.random.choice(temp_route)
            if distance_matrix[current_node, next_node] != np.inf:
                route.append(next_node)
                temp_route.remove(next_node)
                current_node = next_node

                if len(parking) != 0:
                    temp_route = temp_route + parking
                    parking = []

            else:
                parking.append(next_node)
                temp_route.remove(next_node)

            if len(temp_route) == 0 and len(parking) != 0:
                    break

        if len(route) == n:

            if distance_matrix[route[-1], route[0]] != np.inf:
                routes.append(route)

    result = np.array(routes[0]).astype(int)
    
    result = result.tolist()

    return result

# %%
def initialize_population(distance_matrix: np.ndarray, size: int, mutation_rate: float) -> list:

    population = []
    for _ in range(size):
        valid_route = generate_valid_route(distance_matrix)
        population.append(route(valid_route, mutation_rate))

    return population

# %%
def k_tournament(distance_matrix: np.ndarray, population: list, k: int):
    indices = np.random.choice(len(population), k, replace=True)
    sums = [calculate_total_distance(distance_matrix, population[_]['order']) for _ in indices]
    index_shortest_route = indices[np.argmin(sums)]

    return population[index_shortest_route]

# %%
def PMX_crossover(parent1: dict, parent2: dict) -> dict:
    
    p1 = np.array(parent1["order"])
    p2 = np.array(parent2["order"])
    
    try:
        
        mutation_rate = (parent1['mutation_rate'] + parent1['mutation_rate']) / 2
        rng = np.random.default_rng()
        cutoff_1, cutoff_2 = np.sort(rng.choice(np.arange(len(p1)+1), size=2, replace=False))

        offspring = np.zeros(len(p1))
        # Copy the mapping section (middle) from parent1 to offspring
        offspring[cutoff_1:cutoff_2] = p1[cutoff_1:cutoff_2]
        # Map the values in the mapping section from parent2 to offspring
        for i in range(cutoff_1, cutoff_2):
            if p2[i] not in offspring:
                idx = np.where(p2 == p1[i])[0][0]
                
                counter=0
                while offspring[idx] != 0:
                    idx = np.where(p2 == p1[idx])[0][0]
                    counter+=1
                    if counter > 200:
                        return

                offspring[idx] = p2[i]
        for i in range(len(offspring)):
            if offspring[i] == 0:
                offspring[i] = p2[i]

        offspring = offspring.astype(int).tolist()

        return route(offspring, mutation_rate)

    except:
        return

# %%
def mutate(distance_matrix: np.ndarray, original_route: dict) -> dict:
    
    rand = np.random.random()   
    
    if rand <= original_route['mutation_rate']:
        
        valid_route = False
    
        while valid_route is False:
        
            rng = np.random.default_rng()

            mutation = np.array(original_route['order'])

            cutoff_1, cutoff_2 = np.sort(rng.choice(np.arange(len(mutation)+1), size=2, replace=False))

            mutation[cutoff_1:cutoff_2] = np.flip(mutation[cutoff_1:cutoff_2])
            
            mutation = mutation.astype(int).tolist()
            
            total_distance = calculate_total_distance(distance_matrix, mutation)
            
            if total_distance != np.inf:
                valid_route = True
                mutation = route(mutation, original_route['mutation_rate'])
                
        return mutation
    
    else:
        return original_route
    

# %%
def elimination(distance_matrix: dict, population: list, offspring: list, elements_to_keep):

    total = population.copy() + offspring.copy()
    
    total = np.array(total)

    fitnesses = []
    for i in total:
        distance = calculate_total_distance(distance_matrix, i['order'])
        fitnesses.append(distance)
        
    fitnesses = np.array(fitnesses)
    ind = np.argpartition(fitnesses, -elements_to_keep)[:-elements_to_keep]

    new_population = [total[i] for i in ind]

    return new_population

# %%
def evolutionary_algo(distance_matrix: np.ndarray):

    lamda_population = distance_matrix.shape[0]
    mu = lamda_population
    its = 500
    k = 2
    mutation_rate = 0.05
    population = initialize_population(distance_matrix, lamda_population, mutation_rate)

    for iteration in range(its):
        offspring = []
        while len(offspring) != mu:
            parent1 = k_tournament(distance_matrix, population, k)
            parent2 = k_tournament(distance_matrix, population, k)
            new_route = PMX_crossover(parent1, parent2)
            if new_route is not None and calculate_total_distance(distance_matrix, new_route['order']) != np.inf:
                offspring.append(new_route)
                
        offspring = [mutate(distance_matrix, _) for _ in offspring]

        population = [mutate(distance_matrix, _) for _ in population]
        
        population = elimination(distance_matrix, population, offspring, lamda_population)
        
        fitnesses = np.array([calculate_total_distance(distance_matrix, _["order"]) for _ in population])
        print(iteration, f"Mean fitness = {np.mean(fitnesses[np.isfinite(fitnesses)])}, Min fitness = {np.min(fitnesses[np.isfinite(fitnesses)])}")

    return population

tsp = evolutionary_algo(distance_matrix)