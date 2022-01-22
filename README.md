# Stochastic-Heuristics-Optimization

This Python framework is for implementing metaheuristics (or evolutionary computation, or search heuristics).
Its main objective is to concentrate on single operator implementation.

The framework implements a simple sensor placement problem and handle metaheuristics manipulating solutions represented as numerical vectors or bitstrings.

The main interface is implemented in _snp.py_. New algorithms should be integrated within this file and the interface should not be modified. One may add arguments, but not remove or change the contracts of the existing ones.

The file _snp_landscape.py_ is an example that plots the objective function and a greedy search trajectory for a simple problem with only two dimensions.

The design pattern of the framework is a functional approach to composition. The goal is to be able to assemble a metaheuristic, by plugging atomic functions in an algorithm template.

The base of the pattern is a function that contains the main loop of the algorithm, and call other functions called "operators". Example of those algorithms are in the algo module.

For instance, the random algorithm depends on an objective function func, an initialization operator init and a stopping criterion operator again.

Some operator do not depend on the way solutions are encoded (like the stopping criterions) and some operators do depend on the encoding. The former are defined in their own modules while the later are defined in the module corresponding to their encoding (either num or bit).

As they are assembled in an algorithm that do not know their internal in advance, an operators needs to honor an interface. For instance, the init operator's interface takes no input parameter and returns a solution to the problem.

However, some operator may need additional parameters to be passed. To solve this problem, the framework use an interface capture pattern.

There is two ways to capture the interface: either with a functional approach, either with an object-oriented approach. The functional approach enables to implement an operator as a stateless function, and the object-oriented approach is when you need your operator to manage a state.

The functional capture helpers are implemented in the make module. Basically, a function in this module capture the operator function's full interface and returns a function having the expected interface of the operator.

The implicit rule is to use positional arguments for mandatory parameters on which the operator is defined, and keyword arguments for parameters which are specific to the operator.

The object-oriented approach does not need helpers, you just need to define a "functor" class, that is, a class which implements the __call__ interface. This special function member allows to call an instance of a class as if it was a function.

The __call__ method should honor the targeted operator interface. To pass fixed parameters, use the __init__ constructor.

There is an example of an operator implemented this way as the steady class in the _sho/iters.py_ file.

Two example algorithms are provided: a random search and a greedy search. Several useful stopping criterions are provided. The corresponding encoding-dependent operators are also provided, for both numeric and bitstring encodings. The _snp.py_ file shows how to assemble either a numeric greedy solver or a bitstring greedy solver.

To setup your own solver, add your algorithm(s) into the _algo.py_ module, then assemble its instance under its name into _snp.py_. For instance, if you created the annealing algorithm, you will be able to immediatly assemble num_annealing and bit_annealing.

One should be able to call your solvers with _python3 snp.py --solver num_annealing_, for instance.

The _ert.py_ file compute the Expected Run Time Empirical Cumulative Distribution Functions and the plot of the curve.