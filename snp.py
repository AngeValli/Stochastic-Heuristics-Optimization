#encoding: utf-8
import math
import numpy as np
import matplotlib.pyplot as plt

from sho import make, algo, iters, plot, num, bit, pb

########################################################################
# Interface
########################################################################

if __name__=="__main__":
    import argparse

    # Dimension of the search space.
    d = 2

    can = argparse.ArgumentParser()

    can.add_argument("-n", "--nb-sensors", metavar="NB", default=3, type=int,
            help="Number of sensors")

    can.add_argument("-r", "--sensor-range", metavar="RATIO", default=0.3, type=float,
            help="Sensors' range (as a fraction of domain width, max is √2)")

    can.add_argument("-w", "--domain-width", metavar="NB", default=30, type=int,
            help="Domain width (a number of cells)")

    can.add_argument("-i", "--iters", metavar="NB", default=100, type=int,
            help="Maximum number of iterations")

    can.add_argument("-s", "--seed", metavar="VAL", default=None, type=int,
            help="Random pseudo-generator seed (none for current epoch)")

    solvers = ["num_greedy","bit_greedy", "num_annealing", "bit_annealing", "num_evolution", "bit_evolution","num_simple_evolution","bit_simple_evolution","num_dict_evolution"]
    can.add_argument("-m", "--solver", metavar="NAME", choices=solvers, default="num_greedy",
            help="Solver to use, among: "+", ".join(solvers))

    can.add_argument("-t", "--target", metavar="VAL", default=30*30, type=float,
            help="Objective function value target")

    can.add_argument("-y", "--steady-delta", metavar="NB", default=50, type=float,
            help="Stop if no improvement after NB iterations")

    can.add_argument("-e", "--steady-epsilon", metavar="DVAL", default=0, type=float,
            help="Stop if the improvement of the objective function value is lesser than DVAL")

    can.add_argument("-a", "--variation-scale", metavar="RATIO", default=0.3, type=float,
            help="Scale of the variation operators (as a ration of the domain width)")
    
    can.add_argument("-z", "--ert", metavar="BOOL", default=False, type=bool, #This argument is to determine if the snp.py module is executed by the user or the ert.py module. If it is executed by ert.py, we do not show the graphics from snp.py.
            help="Execute without print in shell and without plot")

    the = can.parse_args()

    # Minimum checks.
    assert(0 < the.nb_sensors)
    assert(0 < the.sensor_range <= math.sqrt(2))
    assert(0 < the.domain_width)
    assert(0 < the.iters)

    # Do not forget the seed option,
    # in case you would start "runs" in parallel.
    np.random.seed(the.seed)

    if not the.ert :
    	np.set_printoptions(linewidth = np.inf)


    # Common termination and checkpointing.
    history = []
    iters = make.iter(
                iters.several,
                agains = [
                    make.iter(iters.max,
                        nb_it = the.iters),
                    make.iter(iters.save,
                        filename = the.solver+".csv",
                        fmt = "{it} ; {val} ; {sol}\n"),
                    make.iter(iters.log,
                        fmt="\r{it} {val}"),
                    make.iter(iters.history,
                        history = history),
                    make.iter(iters.target,
                        target = the.target),
                    iters.steady(the.steady_delta, the.steady_epsilon)
                ]
            )

    # Erase the previous file.
    with open(the.solver+".csv", 'w') as fd:
        fd.write("# {} {}\n".format(the.solver,the.domain_width))

    val,sol,sensors = None,None,None
    if the.solver == "num_greedy":
        val,sol = algo.greedy(
                make.func(num.cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range,
                    dim = d * the.nb_sensors),
                make.init(num.rand,
                    dim = d * the.nb_sensors,
                    scale = the.domain_width),
                make.neig(num.neighb_square,
                    scale = the.variation_scale,
                    domain_width = the.domain_width),
                iters
            )
        sensors = num.to_sensors(sol)

    elif the.solver == "bit_greedy":
        val,sol = algo.greedy(
                make.func(bit.cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range,
                    dim = d * the.nb_sensors),
                make.init(bit.rand,
                    domain_width = the.domain_width,
                    nb_sensors = the.nb_sensors),
                make.neig(bit.neighb_square,
                    scale = the.variation_scale,
                    domain_width = the.domain_width),
                iters
            )
        sensors = bit.to_sensors(sol)

    elif the.solver == "num_annealing":
        val,sol = algo.annealing(
                make.func(num.cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range,
                    dim = d * the.nb_sensors),
                make.init(num.rand,
                    dim = d * the.nb_sensors,
                    scale = the.domain_width),
                make.neig(num.neighb_square,
                    scale = the.variation_scale,
                    domain_width = the.domain_width),
                iters,
                temperature0 = 1000,
                Lambda = 0.99
            )
        sensors = num.to_sensors(sol)

    elif the.solver == "bit_annealing":
        val,sol = algo.annealing(
                make.func(bit.cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range,
                    dim = d * the.nb_sensors),
                make.init(bit.rand,
                    domain_width = the.domain_width,
                    nb_sensors = the.nb_sensors),
                make.neig(bit.neighb_square,
                    scale = the.variation_scale,
                    domain_width = the.domain_width),
                iters,
                temperature0 = 1000,
                Lambda = 0.99
            )
        sensors = bit.to_sensors(sol)

    elif the.solver == "num_simple_evolution":
        val,sol = algo.simple_evolution(
                make.func(num.cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range,
                    dim = d * the.nb_sensors),
                make.init(num.rand,
                    dim = d * the.nb_sensors,
                    scale = the.domain_width),
                make.neig(num.neighb_square,
                    scale = the.variation_scale,
                    domain_width = the.domain_width),
                iters,
                n_ech = 2,
                n_pop = 10,
                p_mut = 0.8,
                p_cross = 0.2,
                simple_crossover = num.simple_crossover,
                nb_sensors = the.nb_sensors
            )
        sensors = num.to_sensors(sol)

    elif the.solver == "bit_simple_evolution":
        val,sol = algo.simple_evolution(
                make.func(bit.cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range,
                    dim = d * the.nb_sensors),
                make.init(bit.rand,
                    domain_width = the.domain_width,
                    nb_sensors = the.nb_sensors),
                make.neig(bit.neighb_square,
                    scale = the.variation_scale,
                    domain_width = the.domain_width),
                iters,
                n_ech = 2,
                n_pop = 10,
                p_mut = 0.8,
                p_cross = 0.2,
                simple_crossover = bit.simple_crossover,
                nb_sensors = the.nb_sensors
            )
        sensors = bit.to_sensors(sol)

    elif the.solver == "num_evolution":
        val,sol = algo.evolution(
                make.func(num.cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range,
                    dim = d * the.nb_sensors),
                make.init(num.rand,
                    dim = d * the.nb_sensors,
                    scale = the.domain_width),
                make.neig(num.neighb_square,
                    scale = the.variation_scale,
                    domain_width = the.domain_width),
                iters,
                n_ech = 45,
                n_pop = 100,
                p_mut = 0.8,
                p_cross = 0.2,
                crossover = num.crossover,
                nb_sensors = the.nb_sensors
            )
        sensors = num.to_sensors(sol)

    elif the.solver == "bit_evolution":
        val,sol = algo.evolution(
                make.func(bit.cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range,
                    dim = d * the.nb_sensors),
                make.init(bit.rand,
                    domain_width = the.domain_width,
                    nb_sensors = the.nb_sensors),
                make.neig(bit.neighb_square,
                    scale = the.variation_scale,
                    domain_width = the.domain_width),
                iters,
                n_ech = 45,
                n_pop = 100,
                p_mut = 0.8,
                p_cross = 0.2,
                crossover = bit.crossover,
                nb_sensors = the.nb_sensors
            )
        sensors = bit.to_sensors(sol)

    elif the.solver == "num_dict_evolution":
        val,sol = algo.evolution(
                make.func(num.cover_sum,
                    domain_width = the.domain_width,
                    sensor_range = the.sensor_range,
                    dim = d * the.nb_sensors),
                make.init(num.rand,
                    dim = d * the.nb_sensors,
                    scale = the.domain_width),
                make.neig(num.neighb_square,
                    scale = the.variation_scale,
                    domain_width = the.domain_width),
                iters,
                n_ech = 45,
                n_pop = 100,
                p_mut = 0.8,
                p_cross = 0,
                crossover = num.crossover,
                nb_sensors = the.nb_sensors
            )
        sensors = num.to_sensors(sol)

    if not the.ert :
    	print("\n{} : {}".format(val,sensors))

    shape=(the.domain_width, the.domain_width)

    fig = plt.figure()

    if the.nb_sensors == 1 and the.domain_width <= 50:
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)

        f = make.func(num.cover_sum,
                        domain_width = the.domain_width,
                        sensor_range = the.sensor_range * the.domain_width)
        plot.surface(ax1, shape, f)
        plot.path(ax1, shape, history)
    else:
        ax2=fig.add_subplot(111)

    domain = np.zeros(shape)
    domain = pb.coverage(domain, sensors,
            the.sensor_range * the.domain_width)
    domain = plot.highlight_sensors(domain, sensors)
    ax2.imshow(domain)
    
    if not the.ert :
    	plt.show()