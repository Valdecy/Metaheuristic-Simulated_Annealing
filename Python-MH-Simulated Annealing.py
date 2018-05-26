############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Simulated Annealing

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-Simulated_Annealing, File: Python-MH-Simulated Annealing.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Simulated_Annealing>

############################################################################

# Required Libraries
import pandas as pd
import numpy  as np
import random
import os

# Function: Initialize Variables
def initial_guess(min_values = [-5,-5], max_values = [5,5]):
    n = 1
    guess = pd.DataFrame(np.zeros((n, len(min_values))))
    guess['f(x)'] = 0.0
    for j in range(0, len(min_values)):
         guess.iloc[0,j] = random.uniform(min_values[j], max_values[j])
    guess.iloc[0,-1] = target_function(guess.iloc[0,0:guess.shape[1]-1])
    return guess

# Function: Epson Vector
def epson_vector(guess, mu = 0, sigma = 0.01):
    epson = pd.DataFrame(np.zeros((1, guess.shape[1]-1)))
    for j in range(0, guess.shape[1]-1):
        epson.iloc[0,j] = float(np.random.normal(mu, sigma, 1))
    return epson

# Function: Updtade Solutiion
def update_solution(guess, epson, min_values = [-5,-5], max_values = [5,5]):
    updated_solution = guess.copy(deep = True)
    for j in range(0, guess.shape[1] - 1):
        if (guess.iloc[0,j] + epson.iloc[0,j] > max_values[j]):
            updated_solution.iloc[0,j] = random.uniform(min_values[j], max_values[j])
        elif (guess.iloc[0,j] + epson.iloc[0,j] < min_values[j]):
            updated_solution.iloc[0,j] = random.uniform(min_values[j], max_values[j])
        else:
            updated_solution.iloc[0,j] = guess.iloc[0,j] + epson.iloc[0,j] 
    updated_solution.iloc[0,-1] = target_function(updated_solution.iloc[0,0:updated_solution.shape[1]-1])
    return updated_solution

# SA Function
def simulated_annealing(min_values = [-5,-5], max_values = [5,5], mu = 0, sigma = 0.01, initial_temperature = 1.0, temperature_iterations = 1000, final_temperature = 0.0001, alpha = 0.9):    
    guess = initial_guess(min_values = min_values, max_values = max_values)
    epson = epson_vector(guess, mu = mu, sigma = sigma)
    best  = guess.copy(deep = True)

    fx_best = guess.iloc[0,-1]
    
    Temperature = float(initial_temperature)
    
    while (Temperature > final_temperature):
        
        for repeat in range(0, temperature_iterations):
            print("Temperature = ", Temperature, " ; iteration = ", repeat, " ; f(x) = ", best.iloc[0,-1])
            fx_old  =  guess.iloc[0,-1]
            
            epson     = epson_vector(guess, mu = mu, sigma = sigma)
            new_guess = update_solution(guess, epson, min_values = min_values, max_values = max_values)
            fx_new    = new_guess.iloc[0,-1]
            
            delta = (fx_new - fx_old)
            r = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            p = np.exp(-delta/Temperature)
            
            if (delta < 0 or r <= p):
                guess = new_guess.copy(deep = True)
                
            if (fx_new < fx_best):
                fx_best = fx_new
                best    = guess.copy(deep = True)
        Temperature = alpha*Temperature
        
    print(best)    
    return best

######################## Part 1 - Usage ####################################

# Function to be Minimized. Solution ->  f(x1, x2) = -1.0316; x1 = 0.0898, x2 = -0.7126 or x1 = -0.0898, x2 = 0.7126
def target_function (variables_values = [0, 0]):
    func_value = 4*variables_values[0]**2 - 2.1*variables_values[0]**4 + (1/3)*variables_values[0]**6 + variables_values[0]*variables_values[1] - 4*variables_values[1]**2 + 4*variables_values[1]**4
    return func_value

sa = simulated_annealing(min_values = [-5,-5], max_values = [5,5], mu = 0, sigma = 1, initial_temperature = 1.0, temperature_iterations = 100, final_temperature = 0.0000001, alpha = 0.9)
