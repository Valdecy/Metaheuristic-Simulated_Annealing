# Metaheuristic-Simulated_Annealing
Simulated Annealing to Minimize Functions with Continuous Variables. The function returns: 1) An array containing the used value(s) for the target function and the output of the target function f(x). For example, if the function f(x1, x2) is used, then the array would be [x1, x2, f(x1, x2)].  


* min_values = The minimum value that the variable(s) from a list can have. The default value is -5.

* max_values = The maximum value that the variable(s) from a list can have. The default value is  5.

* target_function = Function to be minimized.

* mu = Mean (of a normal distribution) used to generate a random number. The Default Value is 0.

* sigma = Standard Deviation (of a normal distribution) used to generate a random number. The Default Value is 1.

* initial_temperature = Value to start the simulated annealing. The Default Value is 1.

* temperature_interations = Number of iterations in each temperature. The Default Value is 1000.

* final_temperature = Value to end the simulated annealing. The Default Value is 0.0001.

* alpha = Value that decays the initial_temperature untill the final_temperature. The Default Value is 0.9.
