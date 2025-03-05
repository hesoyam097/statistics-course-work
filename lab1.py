import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, t, norm
import math 


# some constants
GAMMA = 0.01
CONFIDENCE_LEVEL = 0.99 # 1 - GAMMA
N_s = [100, 10_000, 1_000_000]
MEAN = 0
DEV = 1


# help functions
calc_mean = lambda vals, size : np.sum(vals) / size 
calc_sigma_squared = lambda vals, mean, size : np.sum( (vals - mean)**2 ) / size
calc_sigma = lambda vals, mean, size : math.sqrt(calc_sigma_squared(vals, mean, size))
generate_sample = lambda alpha, sigma, size : np.random.normal(alpha, sigma, size)
calc_critical_t_value = lambda confidence, degree: t.ppf((1 + confidence) / 2, degree)



for N in N_s:
    values = generate_sample(MEAN, DEV, N)
    t_value = calc_critical_t_value(CONFIDENCE_LEVEL, N - 1)
    print(t_value)
    mean_values = calc_mean(values, N)
    sigma_values = calc_sigma(values, mean_values, N)

    confidence_interval = (mean_values-t_value * sigma_values/ math.sqrt(N), mean_values+t_value*sigma_values/math.sqrt(N))
    print(confidence_interval)



# print(calc_mean(values))
# print(calc_sigma_squared(values, calc_mean(values)))
# print(calc_sigma(values, calc_mean(values)))
# print(values)

# plt.hist(values, 100)
# plt.axvline(values.mean(), color='k', linestyle='dashed', linewidth=2)
# plt.show()