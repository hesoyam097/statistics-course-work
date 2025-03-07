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


# functions
generate_sample = lambda alpha, sigma, size : np.random.normal(alpha, sigma, size)
calc_mean = lambda vals, size : np.sum(vals) / size 
calc_sigma_squared = lambda vals, mean, size : np.sum( (vals - mean)**2 ) / size
calc_sigma = lambda vals, mean, size : math.sqrt(calc_sigma_squared(vals, mean, size))
calc_critical_t_value = lambda confidence, degree: t.ppf(confidence, degree)
calc_critical_norm_value = lambda confidence: norm.ppf(confidence)
calc_critical_chi2_value = lambda confidence, degree : chi2.ppf(confidence, degree)



def confidence_interval_1(values, N) -> tuple:
    t_value = calc_critical_t_value((1 + CONFIDENCE_LEVEL) / 2, N - 1)
    mean_value = calc_mean(values, N)
    sigma_value = calc_sigma(values, mean_value, N)

    confidence_interval = (mean_value - t_value * sigma_value / math.sqrt(N), mean_value + t_value * sigma_value / math.sqrt(N))
    return confidence_interval


def confidence_interval_2(values, N) -> tuple:
    norm_value = calc_critical_norm_value((1 + CONFIDENCE_LEVEL) / 2)
    mean_value = calc_mean(values, N)
    sigma_value = calc_sigma(values, mean_value, N)

    confidence_interval = (mean_value - norm_value * sigma_value / math.sqrt(N), mean_value + norm_value * sigma_value / math.sqrt(N))
    return confidence_interval


def confidence_interval_3(values, N) -> tuple:
    Z1 = chi2.ppf(CONFIDENCE_LEVEL / 2, N - 1)
    Z2 = chi2.ppf(1 - CONFIDENCE_LEVEL / 2, N - 1) 
    mean_value = calc_mean(values, N)
    sigma_sq_value = calc_sigma_squared(values, mean_value, N)

    confidence_interval = (N * sigma_sq_value / Z2, N * sigma_sq_value / Z1)
    return confidence_interval


for N in N_s:
    values = generate_sample(MEAN, DEV, N)
    
    confidence_interval1 = confidence_interval_1(values, N)
    print(confidence_interval1)
    confidence_interval2 = confidence_interval_2(values, N)
    print(confidence_interval2)
    confidence_interval3 = confidence_interval_3(values, N)
    print(confidence_interval3)
    
    # plt.hist(values, 100)
    # plt.axvline(confidence_interval[0], color='k', linestyle='dashed', linewidth=1)
    # plt.axvline(confidence_interval[1], color='k', linestyle='dashed', linewidth=1)
    # plt.show()


