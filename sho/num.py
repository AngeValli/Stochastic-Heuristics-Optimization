import math
import numpy as np

from . import pb

########################################################################
# Objective functions
########################################################################

# Decoupled from objective functions, so as to be used in display.
def to_sensors(sol):
    """Convert a vector of n*2 dimension to an array of n 2-tuples.

    >>> to_sensors([0,1,2,3])
    [(0, 1), (2, 3)]
    """
    assert(len(sol)>0)
    sensors = []
    for i in range(0,len(sol),2):
        sensors.append( ( int(math.floor(sol[i])), int(math.floor(sol[i+1])) ) )
    return sensors


def cover_sum(sol, domain_width, sensor_range, dim):
    """Compute the coverage quality of the given vector."""
    assert(0 < sensor_range <= domain_width * math.sqrt(2))
    assert(0 < domain_width)
    assert(dim > 0)
    assert(len(sol) >= dim)
    domain = np.zeros((domain_width,domain_width))
    sensors = to_sensors(sol)
    cov = pb.coverage(domain, sensors, sensor_range*domain_width)
    s = np.sum(cov)
    assert(s >= len(sensors))
    return s


########################################################################
# Initialization
########################################################################

def rand(dim, scale):
    """Draw a random vector in [0,scale]**dim."""
    return np.random.random(dim) * scale


########################################################################
# Neighborhood
########################################################################

def neighb_square(sol, scale, domain_width):
    """Draw a random vector in a square of witdh `scale` in [0,1]
    as a fraction of the domain width around the given solution."""
    assert(0 < scale <= 1)
    side = domain_width * scale;
    new = sol + (np.random.random(len(sol)) * side - side/2)
    return new

########################################################################
# Crossovers
########################################################################

def simple_crossover(sol,neighb,j,p_mut,nb_sensors) :
   parent1 = sol[j]
   n = np.random.randint(0,len(sol))
   while n == j :
       n = np.random.randint(0,len(sol))
   parent2 = sol[n]
   
   # We compute the crossover operation by placing the second sensor of parent 2 at the position of second sensor of parent 1.
   child = parent1
   child[2:3] = parent2[2:3]
   
   # Finally, we apply the mutation with probability p_mut
   if np.random.random() < p_mut :
   	child = neighb(child)
   return child

def crossover(sample,j,n_ech,neighb,p_mut,nb_sensors) :

   # We randomly select an element from the sample
   n = np.random.randint(0,n_ech)
   while n == j :
       n = np.random.randint(0,n_ech)
   parent1 = sample[j]
   parent2 = sample[n]
   
   # We compute the crossover operation by placing the second sensor of parent 2 at the position of second sensor of parent 1.
   # The other sensors of parent 1 are in the child population.
   child = parent1
   child[2:3] = parent2[2:3] 		
   
   # Finally, we apply the mutation with probability p_mut
   if np.random.random() < p_mut :
   	child = neighb(child)
   return child

def dict_crossover(dict_sol,sorted_list_keys,ech,key_ech,n_pop,neighb,p_mut,nb_sensors) :
   
   # We randomly select an element from the sample
   n = np.random.randint(0,n_pop)
   sol_key_n = sorted_list_keys[n]
   while sol_key_n == key_ech :
       n = np.random.randint(0,n_pop)
       sol_key_n = sorted_list_keys[n]
   parent1 = ech
   parent2 = dict_sol[sol_key_n][0]
   
   # We compute the crossover operation by placing the second sensor of parent 2 at the position of second sensor of parent 1.
   # The other sensors of parent 1 are in the child population.
   child = parent1
   child[2:3] = parent2[2:3] 		
   
   # Finally, we apply the mutation with probability p_mut
   if np.random.random() < p_mut :
   	child = neighb(child)
   return child

