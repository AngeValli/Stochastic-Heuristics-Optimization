import math
import numpy as np
import copy

from . import x,y,pb

########################################################################
# Objective functions
########################################################################

def cover_sum(sol, domain_width, sensor_range, dim):
    """Compute the coverage quality of the given array of bits."""
    assert(0 < sensor_range <= math.sqrt(2))
    assert(0 < domain_width)
    assert(dim > 0)
    assert(len(sol) >= dim)
    domain = np.zeros((domain_width,domain_width))
    sensors = to_sensors(sol)
    cov = pb.coverage(domain, sensors, sensor_range*domain_width)
    s = np.sum(cov)
    assert(s >= len(sensors))
    return s


def to_sensors(sol):
    """Convert an square array of d lines/columns containing n ones
    to an array of n 2-tuples with related coordinates.

    >>> to_sensors([[1,0],[1,0]])
    [(0, 0), (0, 1)]
    """
    assert(len(sol)>0)
    sensors = []
    for i in range(len(sol)):
        for j in range(len(sol[i])):
            if sol[i][j] == 1:
                sensors.append( (j,i) )
    return sensors


########################################################################
# Initialization
########################################################################

def rand(domain_width, nb_sensors):
    """"Draw a random domain containing nb_sensors ones."""
    domain = np.zeros( (domain_width,domain_width) )
    for x,y in np.random.randint(0, domain_width, (nb_sensors, 2)):
        domain[y][x] = 1
    return domain


########################################################################
# Neighborhood
########################################################################

def neighb_square(sol, scale, domain_width):
    """Draw a random array by moving every ones to adjacent cells."""
    assert(0 < scale <= 1)
    # Copy, because Python pass by reference
    # and we may not want to alter the original solution.
    new = copy.copy(sol)
    for py in range(len(sol)):
        for px in range(len(sol[py])):
            # Indices order is (y,x) in order to match
            # coordinates of images (row,col).
            if sol[py][px] == 1:
                # Add a one somewhere around.
                w = scale/2 * domain_width
                ny = np.random.randint(py-w,py+w)
                nx = np.random.randint(px-w,px+w)
                ny = min(max(0,ny),domain_width-1)
                nx = min(max(0,nx),domain_width-1)

                if new[ny][nx] != 1:
                    new[py][px] = 0 # Remove original position.
                    new[ny][nx] = 1
                # else pass
    return new

########################################################################
# Crossover
########################################################################

def simple_crossover(sol,neighb,j,p_mut,nb_sensors) :
   
   parent1 = sol[j]
   n = np.random.randint(0,len(sol))
   while n == j :
       n = np.random.randint(0,len(sol))
   parent2 = sol[n]
   
   # We compute the crossover operation by placing the second sensor of parent 2 at the position of second sensor of parent 1.
   # The other sensors of parent 1 are in the child population.
   c1 = 0
   child = parent1
   for l in range(len(parent1)) :
   	if c1 == 2 :
   		break
   	else :
	   	for m in range(len(parent1[l])) :
	   		if parent1[l][m] == 1 :
	   			c1 += 1
	   		if c1 == 2 :
	   			child[l][m] = 0
	   			break
  
   # The second sensor of parent 2 are in the child population.
   c2 = 0
   for l in range(len(parent2)) :
   	if c2 == 2 :
   		break
   	else :
	   	for m in range(len(parent2[l])) :
	   		if parent2[l][m] == 1 :
	   			c2 += 1
	   		if c2 == 2 :
	   			child[l][m] = 1
	   			break
   
   if not child.sum() == nb_sensors :
   	child = neighb(parent1)	   				
   
   # Finally, we apply the mutation with probability p_mut
   if np.random.random() < p_mut :
   	child = neighb(child)
   return child

def crossover(sol,j,n_pop,neighb,p_mut,nb_sensors) :
  
   # We compute the crossover operation by placing the second sensor of parent 2 at the position of second sensor of parent 1.
   # The other sensors of parent 1 are in the child population.
   
   # We consider parent 1 as the element of the selected sample.
   parent1 = sol[n_pop+j]
   
   c1 = 0
   child = parent1
   for l in range(len(parent1)) :
   	if c1 == 2 :
   		break
   	else :
	   	for m in range(len(parent1[l])) :
	   		if parent1[l][m] == 1 :
	   			c1 += 1
	   		if c1 == 2 :
	   			child[l][m] = 0
	   			break

   # The second sensor of parent 2 is copied in the child.
   # We select the new parent 2 in the population such as no sensor in the child population is at the position of the second sensor of parent 2
   
   bad_number_sensors = True
   while bad_number_sensors :
	   n = np.random.randint(0,n_pop)
	   while n == n_pop+j :
	   	n = np.random.randint(0,n_pop)
	   parent2 = sol[n]
	   c2 = 0
	   for l in range(len(parent2)) :
	   	if c2 == 2 :
	   		break
	   	else :
		   	for m in range(len(parent2[l])) :
		   		if parent2[l][m] == 1 :
		   			c2 += 1
		   		if c2 == 2 :
		   			child[l][m] = 1
		   			break	

	   if child.sum() == nb_sensors :
	   	bad_number_sensors = False

   # Finally, we apply the mutation with probability p_mut
   if np.random.random() < p_mut :
   	child = neighb(child)
   return child