from scipy.stats import chi2, t, norm
import math
import numpy as np

generate_sample = lambda alpha, sigma, size : np.random.normal(alpha, sigma, size)
calc_mean = lambda vals, size : np.sum(vals) / size 
calc_sigma_squared = lambda vals, mean, size : np.sum( (vals - mean)**2 ) / (size )
calc_sigma = lambda vals, mean, size : math.sqrt(calc_sigma_squared(vals, mean, size))
calc_critical_t_value = lambda confidence, degree: t.ppf(confidence, degree)
calc_critical_norm_value = lambda confidence: norm.ppf(confidence)
calc_critical_chi2_value = lambda confidence, degree : chi2.ppf(confidence, degree)