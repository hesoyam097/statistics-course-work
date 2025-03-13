import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math

# Встановлюємо рівень значимості
alpha = 0.05

# Завдання 1: Перевірка гіпотези однорідності (критерій пустих блоків)
def empty_blocks_test(X, Y, m, k):
    """
    Критерій пустих блоків для перевірки гіпотези однорідності
    X, Y - вибірки
    m - кількість блоків
    k - розмір блоку
    """
    # Об'єднуємо вибірки
    Z = np.concatenate((X, Y))
    n = len(Z)

    # Сортуємо об'єднану вибірку
    Z_sorted = np.sort(Z)

    # Розбиваємо на m блоків
    blocks = np.array_split(Z_sorted, m)

    # Рахуємо кількість пустих блоків (блоків без елементів з Y)
    mu = 0
    for block in blocks:
        # Блок вважається пустим, якщо в ньому немає елементів з Y
        if not any(np.isin(block, Y)):
            mu += 1

    # Обчислюємо статистику критерію
    n1 = len(X)
    n2 = len(Y)
    p = n1 / n
    q = n2 / n

    # Математичне сподівання та дисперсія для mu при справедливості H0
    E_mu = m * (p ** k)
    D_mu = m * (p ** k) * (1 - p ** k)

    # Нормована статистика
    if D_mu > 0:
        U = (mu - E_mu) / np.sqrt(D_mu)
    else:
        U = 0

    # Критичне значення для двостороннього тесту
    critical_value = stats.norm.ppf(1 - alpha/2)

    # Перевірка гіпотези
    if abs(U) > critical_value:
        result = "Відхиляємо H0 (вибірки не однорідні)"
    else:
        result = "Не відхиляємо H0 (вибірки однорідні)"

    return {
        "mu": mu,
        "E_mu": E_mu,
        "D_mu": D_mu,
        "U": U,
        "critical_value": critical_value,
        "result": result
    }

# Завдання 2: Перевірка гіпотези незалежності
def generate_dependent_sample(n, case):
    """
    Генерує вибірку (X, Y) за заданим правилом
    """
    X = np.random.uniform(0, 1, n)

    if case == 'a':
        # Випадок а): Y = X + eps, eps ~ U(0, 0.2)
        eps = np.random.uniform(0, 0.2, n)
        Y = X + eps
        Y = np.minimum(Y, 1)  # Нормалізуємо Y до [0, 1]
    elif case == 'b':
        # Випадок b): Y = X^2 + eps, eps ~ U(0, 0.2)
        eps = np.random.uniform(0, 0.2, n)
        Y = X**2 + eps
        Y = np.minimum(Y, 1)  # Нормалізуємо Y до [0, 1]

    return X, Y

def spearman_test(X, Y):
    """
    Критерій Спірмена для перевірки гіпотези незалежності
    """
    n = len(X)

    # Обчислюємо коефіцієнт кореляції Спірмена
    rho, p_value = stats.spearmanr(X, Y)

    # Статистика критерію
    T = rho * np.sqrt(n - 2) / np.sqrt(1 - rho**2)

    # Критичне значення для двостороннього тесту
    critical_value = stats.t.ppf(1 - alpha/2, n - 2)

    # Перевірка гіпотези
    if abs(T) > critical_value:
        result = "Відхиляємо H0 (змінні залежні)"
    else:
        result = "Не відхиляємо H0 (змінні незалежні)"

    return {
        "rho": rho,
        "T": T,
        "critical_value": critical_value,
        "p_value": p_value,
        "result": result
    }

def kendall_test(X, Y):
    """
    Критерій Кендалла для перевірки гіпотези незалежності
    """
    n = len(X)

    # Обчислюємо коефіцієнт кореляції Кендалла
    tau, p_value = stats.kendalltau(X, Y)

    # Статистика критерію
    sigma_tau = np.sqrt((2 * (2*n + 5)) / (9 * n * (n - 1)))
    Z = tau / sigma_tau

    # Критичне значення для двостороннього тесту
    critical_value = stats.norm.ppf(1 - alpha/2)

    # Перевірка гіпотези
    if abs(Z) > critical_value:
        result = "Відхиляємо H0 (змінні залежні)"
    else:
        result = "Не відхиляємо H0 (змінні незалежні)"

    return {
        "tau": tau,
        "Z": Z,
        "critical_value": critical_value,
        "p_value": p_value,
        "result": result
    }

# Завдання 3: Перевірка гіпотези випадковості
def generate_sample_task3(n, a):
    """
    Генерує вибірку за правилом X_i = a*X_{i-1} + (1-a)*eps_i
    """
    X = np.zeros(n)
    eps = np.random.uniform(0, 1, n)

    # Початкове значення
    X[0] = eps[0]

    # Генеруємо послідовність
    for i in range(1, n):
        X[i] = a * X[i-1] + (1-a) * eps[i]

    return X

def count_inversions(X):
    """
    Підраховує кількість інверсій у послідовності
    """
    n = len(X)
    inv_count = 0

    for i in range(n):
        for j in range(i+1, n):
            if X[i] > X[j]:
                inv_count += 1

    return inv_count

def inversion_test(X):
    """
    Критерій випадковості на основі кількості інверсій
    """
    n = len(X)

    # Підраховуємо кількість інверсій
    A = count_inversions(X)

    # Математичне сподівання та дисперсія для A при справедливості H0
    E_A = n * (n - 1) / 4
    D_A = n * (n - 1) * (2*n + 5) / 72

    # Нормована статистика
    Z = (A - E_A) / np.sqrt(D_A)

    # Критичне значення для двостороннього тесту
    critical_value = stats.norm.ppf(1 - alpha/2)

    # Перевірка гіпотези
    if abs(Z) > critical_value:
        result = "Відхиляємо H0 (послідовність не випадкова)"
    else:
        result = "Не відхиляємо H0 (послідовність випадкова)"

    return {
        "A": A,
        "E_A": E_A,
        "D_A": D_A,
        "Z": Z,
        "critical_value": critical_value,
        "result": result
    }