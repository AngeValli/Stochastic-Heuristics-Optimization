########################################################################
# Algorithms
########################################################################
import numpy as np

def random(func, init, again):
    """Iterative random search template."""
    best_sol = init()
    best_val = func(best_sol)
    val,sol = best_val,best_sol
    i = 0
    while again(i, best_val, best_sol):
        sol = init()
        val = func(sol)
        if val >= best_val:
            best_val = val
            best_sol = sol
        i += 1
    return best_val, best_sol


def greedy(func, init, neighb, again):
    """Iterative randomized greedy heuristic template."""
    best_sol = init()
    best_val = func(best_sol)
    val,sol = best_val,best_sol
    i = 1
    while again(i, best_val, best_sol):
        sol = neighb(best_sol)
        val = func(sol)
        # Use >= and not >, so as to avoid random walk on plateus.
        if val >= best_val:
            best_val = val
            best_sol = sol
        i += 1
    return best_val, best_sol

def annealing(func, init, neighb, again, temperature0, Lambda) :
    best_sol = init()
    temperature = temperature0
    best_val = func(best_sol)
    val, sol = best_val, best_sol
    i = 1
    while again(i, best_val, best_sol) :
    	sol = neighb(best_sol)
    	val = func(sol)
    	if val >= best_val and np.exp((best_val-val)/temperature) > np.random.random() :
    		best_sol = sol
    		best_val = val
    	temperature = Lambda*temperature
    	i += 1
    return best_val, best_sol

# Genetic algorithm with sample of the n_ech best elements of the population

def simple_evolution(func, init, neighb, again, n_pop, n_ech, p_mut, p_cross, simple_crossover, nb_sensors) :
    sol = []
    for _ in range(n_pop) :
    	sol.append(init())
    sol = sorted(sol,key=func,reverse=True)
    best_sol = sol[0]
    best_val = func(best_sol)
    i = 1
    while again(i, best_val, best_sol):
    	for j in range(n_ech) :
    		if np.random.random() < p_mut :
    			sol.append(neighb(sol[j]))
    		elif np.random.random() < p_cross:
    			sol.append(simple_crossover(sol[:n_ech],neighb,j,p_mut,nb_sensors))	
    	sol = sorted(sol,key=func,reverse=True)
    	sol = sol[:n_pop-n_ech]
    	best_sol = sol[0]
    	best_val = func(best_sol)
    	i += 1
    return best_val, best_sol


# Genetic algorithm with random sample of the population

def evolution(func, init, neighb, again, n_pop, n_ech, p_mut, p_cross, crossover, nb_sensors) :
    
    # We initialize a vector of the population of size n_pop and decreasing order on the values, so the best candidate is at first position.
    sol = []
    for _ in range(n_pop) :
    	sol.append(init())
    sol = sorted(sol,key=func,reverse=True)
    best_sol = sol[0]
    best_val = func(best_sol)
    i = 1
    
    # We use a sample of size n_ech
    while again(i, best_val, best_sol):
    	
    	# We randomly choose two solutions in the population and we keep the best in the sample. We append it at the end of the vector.
    	for j in range(n_ech) :
    		ech1 = np.random.randint(0,n_pop)
    		ech2 = np.random.randint(0,n_pop)
    		while ech1 == ech2 :
    			ech2 = np.random.randint(0,n_pop)
    		if func(sol[ech2]) > func(sol[ech1]) :
    			sol.append(sol[ech2])
    		else :
    			sol.append(sol[ech1])
    		
    		# We apply the mutation operator (function neighb) with a probability p_mut and the crossover operation with a probability p_cross to all elements of the sample.
	    	if np.random.random() < p_mut :
	    		sol[n_pop+j] = neighb(sol[n_pop+j])
	    	
	    	elif np.random.random() < p_cross:
	    		sol[n_pop+j] = crossover(sol,j,n_pop,neighb,p_mut,nb_sensors)
    	
    	# Finally, we compute the new population by eliminating the n_ech worst values.
    	sol = sorted(sol,key=func,reverse=True)
    	sol = sol[:n_pop]
    	best_sol = sol[0]
    	best_val = func(best_sol)
    	i += 1
    return best_val, best_sol

# We compute the genetic algorithm with random sampling and using a dictionary, so we keep in memory the selected samples. We ensure to select only different elements at one iteration.

def dict_evolution(func, init, neighb, again, n_pop, n_ech, p_mut, p_cross, crossover, nb_sensors) :
    
    # We initialize a vector of the population of size n_pop and decreasing order on the values, so the best candidate is at first position.
    dict_sol = dict()
    i = 0
    for _ in range(n_pop) :
    	i+=1
    	sol = init()
    	val = func(sol)
    	dict_sol[i] = (sol,val)
    sorted_dict_sol = {k: v for k, v in sorted(dict_sol.items(), key=lambda item: item[1][1])}
    best_val = list(sorted_dict_sol.values())[-1][1]
    best_sol = list(sorted_dict_sol.values())[-1][0]
    i = 1
    count_appel_fonction_obj = n_pop

    # We use a sample of size n_ech
    while again(i, best_val, best_sol):
    	
    	liste_indice_tire = []
    	sorted_list_keys = list(sorted_dict_sol.keys())
    	
    	
    	# We randomly choose two elements in the population and we keep the best one in the sample.
    	for j in range(n_ech) :
    	
    		n_ech1 = np.random.randint(0,n_pop)
    		while n_ech1 in liste_indice_tire :
    			n_ech1 = np.random.randint(0,n_pop)
    		n_ech2 = np.random.randint(0,n_pop)
    		while n_ech1 == n_ech2 or n_ech2 in liste_indice_tire :
    			n_ech2 = np.random.randint(0,n_pop)
    		
    		if n_ech1 > n_ech2 :
    			key_ech = sorted_list_keys[n_ech1]
    			liste_indice_tire.append(n_ech1)
    		else :
    			key_ech = sorted_list_keys[n_ech2]
    			liste_indice_tire.append(n_ech2)

    		ech = dict_sol[key_ech][0]

    		# We apply the mutation operator (function neighb) with a probability p_mut and the crossover operation with a probability p_cross to all elements of the sample.
	    	if np.random.random() < p_mut :
	    		sol = neighb(ech)
	    		val = func(sol)
	    		dict_sol[key_ech] = (sol,val)
	    		count_appel_fonction_obj += 1
	    	
	    	elif np.random.random() < p_cross:
	    		sol = crossover(dict_sol,sorted_list_keys,ech,key_ech,n_pop,neighb,p_mut,nb_sensors)
	    		val = func(sol)
	    		dict_sol[key_ech] = (sol,val)
	    		count_appel_fonction_obj += 1

    	# Finally, we implement the new population
    	sorted_dict_sol = {k: v for k, v in sorted(dict_sol.items(), key=lambda item: item[1][1])}
    	best_val = list(sorted_dict_sol.values())[-1][1]
    	best_sol = list(sorted_dict_sol.values())[-1][0]
    	i += 1
    return best_val, best_sol
