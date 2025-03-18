
import matplotlib.pyplot as plt
from funcs import *


# some constants
GAMMA = 0.01
CONFIDENCE_LEVEL = 0.99 # 1 - GAMMA
N_s = [100, 10_000, 1_000_000]
MEAN = 0
DEV = 1


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
    Z1 = calc_critical_chi2_value((1-CONFIDENCE_LEVEL) / 2, N - 1)
    Z2 = calc_critical_chi2_value((1+CONFIDENCE_LEVEL) / 2, N - 1)
    mean_value = calc_mean(values, N)
    sigma_sq_value = calc_sigma_squared(values, mean_value, N)

    confidence_interval = (N * sigma_sq_value / Z2, N * sigma_sq_value / Z1)
    return confidence_interval


def display_results(title, interval, values, N):
    """Prints formatted output and displays histogram with confidence intervals."""
    print("=" * 50)
    print(f"{title}")
    print("=" * 50)
    print(f"📊 Розмір вибірки (N): {N}")
    print(f"📈 Середнє значення: {calc_mean(values, N):.5f}")
    print(f"🔹 Інтервал довіри: ({interval[0]:.5f}, {interval[1]:.5f})")
    print(f"📏 Довжина інтервалу: {interval[1] - interval[0]:.5f}")
    print("-" * 50)

    plt.hist(values, bins=100, alpha=0.75, color="skyblue", edgecolor="black")
    plt.axvline(interval[0], color='red', linestyle='dashed', linewidth=2, label="Lower Bound")
    plt.axvline(interval[1], color='green', linestyle='dashed', linewidth=2, label="Upper Bound")
    plt.legend()
    plt.title(title)
    plt.xlabel("Значення")
    plt.ylabel("Частота")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


# Running the calculations
for N in N_s:
    values = generate_sample(MEAN, DEV, N)

    display_results("📏 Довірчий інтервал для середнього", 
                    confidence_interval_1(values, N), values, N)

    display_results("📏 Довірчий інтервал для середнього", 
                    confidence_interval_2(values, N), values, N)

    display_results("📏 Довірчий інтервал для дисперсії", 
                    confidence_interval_3(values, N), values, N)