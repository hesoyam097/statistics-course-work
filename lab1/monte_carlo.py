from funcs import *
import scipy.stats as stats
from scipy.special import gamma

def exact_Q(alpha):
    return alpha**4 * gamma(3)

def monte_carlo_method(alpha, n):
    omega_xi = -np.log(np.random.uniform(size=n)) / alpha
    omega_eta = -np.log(np.random.uniform(size=n))
    q_hat = np.mean(omega_xi < omega_eta)
    variance = np.var(omega_xi < omega_eta, ddof=1) / n
    return q_hat, variance

def method_2(alpha, n):
    omega_xi = -np.log(np.random.uniform(size=n)) / alpha
    q_hat = np.mean(np.exp(-omega_xi**2))
    variance = np.var(np.exp(-omega_xi**2), ddof=1) / n
    return q_hat, variance

def method_3(alpha, n):
    omega_eta = -np.log(np.random.uniform(size=n))
    q_hat = np.mean(1 - np.exp(-alpha * omega_eta))
    variance = np.var(1 - np.exp(-alpha * omega_eta), ddof=1) / n
    return q_hat, variance

def method_4(alpha, n):
    omega_gamma = np.random.gamma(shape=2, scale=1, size=n)
    q_hat = np.mean(1 - np.exp(-alpha * omega_gamma))
    variance = np.var(1 - np.exp(-alpha * omega_gamma), ddof=1) / n
    return q_hat, variance

def compute_confidence_interval(q_hat, variance, n, confidence=0.99):
    z_value = stats.norm.ppf(1 - (1 - confidence) / 2)
    margin = z_value * np.sqrt(variance)
    return q_hat - margin, q_hat + margin

def required_realizations(variance, q_hat, confidence=0.99, epsilon=0.01):
    z_value = stats.norm.ppf(1 - (1 - confidence) / 2)
    return int(np.ceil((z_value**2 * variance) / (epsilon**2 * q_hat**2)))

alphas = [1, 0.3, 0.1]
n_initial = 1000  # начальное число реализаций

for alpha in alphas:
    print(f"\nResults for alpha = {alpha}")
    q_exact = exact_Q(alpha)
    print(f"Exact Q(a): {q_exact:.6f}")
    
    for method, func in zip(["Monte Carlo", "Method 2", "Method 3", "Method 4"],
                             [monte_carlo_method, method_2, method_3, method_4]):
        q_hat, variance = func(alpha, n_initial)
        ci_lower, ci_upper = compute_confidence_interval(q_hat, variance, n_initial)
        n_required = required_realizations(variance, q_hat)
        
        print(f"\n{method}:")
        print(f"Estimated Q(a): {q_hat:.6f}")
        print(f"Sample Variance: {variance:.6e}")
        print(f"Confidence Interval (99%): ({ci_lower:.6f}, {ci_upper:.6f})")
        print(f"Required Realizations: {n_required}")
