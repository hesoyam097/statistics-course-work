import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from math import sqrt, gamma

#Довірчі інтервали

def generate_samples(n):

    U1 = np.random.uniform(0, 1, n)
    U2 = np.random.uniform(0, 1, n)

    # Бокса-Мюллера
    Z1 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    Z2 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)

    # Обчислюємо X = Z1^2 + Z2^2 (розподіл хі-квадрат з 2 ступенями свободи)
    X = Z1**2 + Z2**2

    return X

def confidence_interval_mean_normal(sample, confidence=0.95):
    n = len(sample)
    mean = np.mean(sample)
    std_dev = np.std(sample, ddof=1)  # Незміщена оцінка стандартного відхилення

    # Критичне значення t-розподілу
    t_critical = stats.t.ppf((1 + confidence) / 2, df=n-1)

    margin_error = t_critical * (std_dev / sqrt(n))

    lower_bound = mean - margin_error
    upper_bound = mean + margin_error

    return mean, (lower_bound, upper_bound), upper_bound - lower_bound

def confidence_interval_mean_unknown(sample, confidence=0.95):
    n = len(sample)
    mean = np.mean(sample)
    std_dev = np.std(sample, ddof=1)

    # Критичне значення нормального розподілу (використовуючи ЦГТ)
    z_critical = stats.norm.ppf((1 + confidence) / 2)

    margin_error = z_critical * (std_dev / sqrt(n))

    lower_bound = mean - margin_error
    upper_bound = mean + margin_error

    return mean, (lower_bound, upper_bound), upper_bound - lower_bound

def confidence_interval_variance_normal(sample, confidence=0.95):
    n = len(sample)
    variance = np.var(sample, ddof=1)  

    # Критичні значення розподілу хі-квадрат
    chi2_lower = stats.chi2.ppf((1 - confidence) / 2, df=n-1)
    chi2_upper = stats.chi2.ppf((1 + confidence) / 2, df=n-1)

    lower_bound = (n - 1) * variance / chi2_upper
    upper_bound = (n - 1) * variance / chi2_lower

    return variance, (lower_bound, upper_bound), upper_bound - lower_bound

def task1():
    # Розміри вибірок для тестування
    sample_sizes = [100, 1000, 10000]
    confidence_levels = [0.9, 0.95, 0.99]

    # Справжні параметри (для хі-квадрат з 2 ступенями свободи: мат. сподівання = 2, дисперсія = 4)
    true_mean = 2
    true_variance = 4

    print("Завдання 1: Довірчі інтервали")
    print("=" * 80)

    for confidence in confidence_levels:
        print(f"\nРівень довіри: {confidence}")
        print("-" * 60)

        for n in sample_sizes:
            print(f"\nРозмір вибірки: {n}")

            # Генеруємо вибірку
            sample = generate_samples(n)

            # a) Довірчий інтервал для мат. сподівання (нормальний розподіл, невідома дисперсія)
            mean_est, mean_ci, mean_width = confidence_interval_mean_normal(sample, confidence)
            print("\na) Мат. сподівання (нормальний розподіл, невідома дисперсія):")
            print(f"   Кількість реалізацій: {n}")
            print(f"   Оцінка мат. сподівання: {mean_est:.6f}")
            print(f"   Довірчий інтервал: ({mean_ci[0]:.6f}, {mean_ci[1]:.6f})")
            print(f"   Ширина інтервалу: {mean_width:.6f}")
            print(f"   Справжнє мат. сподівання {true_mean} {('входить в інтервал' if mean_ci[0] <= true_mean <= mean_ci[1] else 'не входить в інтервал')}")

            # b) Довірчий інтервал для мат. сподівання (невідомий розподіл)
            mean_est, mean_ci, mean_width = confidence_interval_mean_unknown(sample, confidence)
            print("\nb) Мат. сподівання (невідомий розподіл):")
            print(f"   Кількість реалізацій: {n}")
            print(f"   Оцінка мат. сподівання: {mean_est:.6f}")
            print(f"   Довірчий інтервал: ({mean_ci[0]:.6f}, {mean_ci[1]:.6f})")
            print(f"   Ширина інтервалу: {mean_width:.6f}")
            print(f"   Справжнє мат. сподівання {true_mean} {('входить в інтервал' if mean_ci[0] <= true_mean <= mean_ci[1] else 'не входить в інтервал')}")

            # c) Довірчий інтервал для дисперсії (нормальний розподіл)
            var_est, var_ci, var_width = confidence_interval_variance_normal(sample, confidence)
            print("\nc) Дисперсія (нормальний розподіл):")
            print(f"   Кількість реалізацій: {n}")
            print(f"   Оцінка дисперсії: {var_est:.6f}")
            print(f"   Довірчий інтервал: ({var_ci[0]:.6f}, {var_ci[1]:.6f})")
            print(f"   Ширина інтервалу: {var_width:.6f}")
            print(f"   Справжня дисперсія {true_variance} {('входить в інтервал' if var_ci[0] <= true_variance <= var_ci[1] else 'не входить в інтервал')}")

# ====================== Завдання 2: Обчислення ймовірності ======================

def exact_probability(alpha):
    return alpha / (1 + alpha)

def method1(alpha, n):
    # Генеруємо X ~ Gamma(2, 1)
    X = np.random.gamma(shape=2, scale=1, size=n)

    # Генеруємо Y ~ Gamma(2, 1/alpha)
    Y = np.random.gamma(shape=2, scale=1/alpha, size=n)

    # Обчислюємо індикаторну функцію I(X > Y)
    indicators = (X > Y).astype(int)

    # Оцінюємо ймовірність
    p_hat = np.mean(indicators)

    # Оцінюємо дисперсію
    var_hat = np.var(indicators, ddof=1)

    return p_hat, var_hat, indicators

def method2(alpha, n):
    # Генеруємо X ~ Gamma(2, 1)
    X = np.random.gamma(shape=2, scale=1, size=n)

    # Генеруємо U ~ Uniform(0, 1)
    U = np.random.uniform(0, 1, n)

    # Обчислюємо F_Y(X) = 1 - (1 + alpha*X)*exp(-alpha*X)
    F_Y_X = 1 - (1 + alpha*X) * np.exp(-alpha*X)

    # Обчислюємо індикаторну функцію I(U > F_Y(X))
    indicators = (U > F_Y_X).astype(int)

    # Оцінюємо ймовірність
    p_hat = np.mean(indicators)

    # Оцінюємо дисперсію
    var_hat = np.var(indicators, ddof=1)

    return p_hat, var_hat, indicators

def method3(alpha, n):
    # Генеруємо X ~ Gamma(2, 1)
    X = np.random.gamma(shape=2, scale=1, size=n)

    # Обчислюємо 1 - F_Y(X) = (1 + alpha*X)*exp(-alpha*X)
    values = (1 + alpha*X) * np.exp(-alpha*X)

    # Оцінюємо ймовірність
    p_hat = np.mean(values)

    # Оцінюємо дисперсію
    var_hat = np.var(values, ddof=1)

    return p_hat, var_hat, values

def method4(alpha, n):
    # Генеруємо Z1, Z2 ~ Exponential(1)
    Z1 = np.random.exponential(scale=1, size=n)
    Z2 = np.random.exponential(scale=1, size=n)

    # Обчислюємо Z = Z1 + Z2 (що має розподіл Gamma(2, 1))
    Z = Z1 + Z2

    # Обчислюємо h(Z) = (1 + alpha*Z)*exp(-alpha*Z)
    h_Z = (1 + alpha*Z) * np.exp(-alpha*Z)

    # Обчислюємо g(Z) = Z*exp(-Z) (щільність Gamma(2, 1))
    g_Z = Z * np.exp(-Z)

    # Обчислюємо f(Z) = h(Z) / g(Z)
    values = h_Z / g_Z

    # Оцінюємо ймовірність
    p_hat = np.mean(values)

    # Оцінюємо дисперсію
    var_hat = np.var(values, ddof=1)

    return p_hat, var_hat, values

def calculate_required_n(var_hat, p_hat, confidence=0.99, rel_error=0.01):
    # Початкова кількість реалізацій для стабілізації
    n0 = 1000

    # Обчислюємо z-значення для заданого рівня довіри
    z = stats.norm.ppf((1 + confidence) / 2)

    # Обчислюємо необхідну кількість реалізацій
    n_required = max(n0, int(np.ceil((z**2 * var_hat) / ((rel_error * p_hat)**2))))

    return n_required

def task2():
    # Параметри
    alpha_values = [1.0, 0.3, 0.1]
    confidence = 0.99
    rel_error = 0.01
    initial_n = 10000  # Початковий розмір вибірки для оцінки

    print("\nЗавдання 2: Обчислення ймовірності")
    print("=" * 80)

    for alpha in alpha_values:
        print(f"\nПараметр alpha = {alpha}")
        print("-" * 60)

        # Обчислюємо точну ймовірність
        exact_prob = exact_probability(alpha)
        print(f"Точна ймовірність P(X > Y) = {exact_prob:.6f}")

        # Метод 1: Стандартний метод Монте-Карло
        p_hat1, var_hat1, _ = method1(alpha, initial_n)
        n_required1 = calculate_required_n(var_hat1, p_hat1, confidence, rel_error)

        # Перераховуємо з необхідним розміром вибірки
        if n_required1 > initial_n:
            p_hat1, var_hat1, _ = method1(alpha, n_required1)

        # Обчислюємо довірчий інтервал
        z = stats.norm.ppf((1 + confidence) / 2)
        margin_error1 = z * sqrt(var_hat1 / n_required1)
        ci1 = (p_hat1 - margin_error1, p_hat1 + margin_error1)

        print("\nМетод 1 (Стандартний метод Монте-Карло):")
        print(f"   Оцінка ймовірності: {p_hat1:.6f}")
        print(f"   Вибіркова дисперсія: {var_hat1:.6f}")
        print(f"   Довірчий інтервал: ({ci1[0]:.6f}, {ci1[1]:.6f})")
        print(f"   Необхідна кількість реалізацій: {n_required1}")

        # Метод 2
        p_hat2, var_hat2, _ = method2(alpha, initial_n)
        n_required2 = calculate_required_n(var_hat2, p_hat2, confidence, rel_error)

        # Перераховуємо з необхідним розміром вибірки
        if n_required2 > initial_n:
            p_hat2, var_hat2, _ = method2(alpha, n_required2)

        # Обчислюємо довірчий інтервал
        margin_error2 = z * sqrt(var_hat2 / n_required2)
        ci2 = (p_hat2 - margin_error2, p_hat2 + margin_error2)

        print("\nМетод 2:")
        print(f"   Оцінка ймовірності: {p_hat2:.6f}")
        print(f"   Вибіркова дисперсія: {var_hat2:.6f}")
        print(f"   Довірчий інтервал: ({ci2[0]:.6f}, {ci2[1]:.6f})")
        print(f"   Необхідна кількість реалізацій: {n_required2}")

        # Метод 3
        p_hat3, var_hat3, _ = method3(alpha, initial_n)
        n_required3 = calculate_required_n(var_hat3, p_hat3, confidence, rel_error)

        # Перераховуємо з необхідним розміром вибірки
        if n_required3 > initial_n:
            p_hat3, var_hat3, _ = method3(alpha, n_required3)

        # Обчислюємо довірчий інтервал
        margin_error3 = z * sqrt(var_hat3 / n_required3)
        ci3 = (p_hat3 - margin_error3, p_hat3 + margin_error3)

        print("\nМетод 3:")
        print(f"   Оцінка ймовірності: {p_hat3:.6f}")
        print(f"   Вибіркова дисперсія: {var_hat3:.6f}")
        print(f"   Довірчий інтервал: ({ci3[0]:.6f}, {ci3[1]:.6f})")
        print(f"   Необхідна кількість реалізацій: {n_required3}")

        # Метод 4
        p_hat4, var_hat4, _ = method4(alpha, initial_n)
        n_required4 = calculate_required_n(var_hat4, p_hat4, confidence, rel_error)

        # Перераховуємо з необхідним розміром вибірки
        if n_required4 > initial_n:
            p_hat4, var_hat4, _ = method4(alpha, n_required4)

        # Обчислюємо довірчий інтервал
        margin_error4 = z * sqrt(var_hat4 / n_required4)
        ci4 = (p_hat4 - margin_error4, p_hat4 + margin_error4)

        print("\nМетод 4:")
        print(f"   Оцінка ймовірності: {p_hat4:.6f}")
        print(f"   Вибіркова дисперсія: {var_hat4:.6f}")
        print(f"   Довірчий інтервал: ({ci4[0]:.6f}, {ci4[1]:.6f})")
        print(f"   Необхідна кількість реалізацій: {n_required4}")

        # Порівнюємо методи
        print("\nПорівняння методів (необхідна кількість реалізацій):")
        methods = {
            "Метод 1 (Стандартний метод Монте-Карло)": n_required1,
            "Метод 2": n_required2,
            "Метод 3": n_required3,
            "Метод 4": n_required4
        }

        # Сортуємо методи за необхідною кількістю реалізацій
        sorted_methods = sorted(methods.items(), key=lambda x: x[1])

        for i, (method_name, n_req) in enumerate(sorted_methods):
            print(f"   {i+1}. {method_name}: {n_req}")

# Запускаємо обидва завдання
if __name__ == "__main__":
    task1()
    task2()