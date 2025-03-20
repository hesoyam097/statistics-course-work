import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math

# Встановлюємо рівень значимості
alpha = 0.05

# Завдання 1: Перевірка гіпотези однорідності (критерій пустих блоків)
def empty_blocks_test(n, m):
    """
    Performs the empty blocks test for homogeneity.
    
    Parameters:
    n - sample size for X (Exp(1))
    m - sample size for Y (Exp(1.2))
    """
    # Generate samples
    X = np.random.exponential(scale=1.0, size=n)     # Exp(1)
    Y = np.random.exponential(scale=1/1.2, size=m)   # Exp(1.2)

    # Merge and sort the samples
    Z = np.concatenate((X, Y))
    Z_sorted = np.sort(Z)

    # Divide into m blocks
    blocks = np.array_split(Z_sorted, m)

    # Count empty blocks (blocks that contain only X values)
    empty_count = sum(all(val in X for val in block) for block in blocks)

    # Compute test statistic
    p = n / (n + m)
    expected_empty = m * (p ** (n / m))
    variance_empty = m * (p ** (n / m)) * (1 - p ** (n / m))
    
    if variance_empty > 0:
        U = (empty_count - expected_empty) / np.sqrt(variance_empty)
    else:
        U = 0

    # Critical value for a two-tailed test
    critical_value = stats.norm.ppf(1 - alpha / 2)

    # Hypothesis test result
    if abs(U) > critical_value:
        result = "Reject H0 (samples are not homogeneous)"
    else:
        result = "Fail to reject H0 (samples are homogeneous)"

    return {
        "Empty Blocks Count": empty_count,
        "Expected Empty Blocks": expected_empty,
        "Variance": variance_empty,
        "Test Statistic U": U,
        "Critical Value": critical_value,
        "Result": result
    }

# Завдання 2: Перевірка гіпотези незалежності
def generate_sample(n, case):
    """
    Генерує вибірку (X, Y) за заданим правилом відповідно до завдання.
    
    Параметри:
    n - розмір вибірки
    case - 'a' або 'b', що визначає метод генерації Y
    """
    X = np.random.uniform(0, 1, n)  # X ~ U(0,1)
    eta = np.random.uniform(-1, 1, n)  # eta ~ U(-1,1)

    if case == 'a':
        Y = X * eta  # Випадок (а): Y = X * η
    elif case == 'b':
        Y = X + eta  # Випадок (б): Y = X + η

    return X, Y

def spearman_rank_correlation(X, Y):
    """
    Обчислення коефіцієнта кореляції Спірмена вручну.
    """
    n = len(X)

    # Ранжування X та Y
    rank_X = np.argsort(np.argsort(X)) + 1
    rank_Y = np.argsort(np.argsort(Y)) + 1

    # Обчислення різниці рангів
    d = rank_X - rank_Y

    # Обчислення коефіцієнта Спірмена
    rho = 1 - (6 * np.sum(d**2)) / (n * (n**2 - 1))

    # Статистика Спірмена
    T = rho * np.sqrt(n - 2) / np.sqrt(1 - rho**2)
    critical_value = stats.norm.ppf(1 - alpha / 2)

    result = "Відхиляємо H0 (змінні залежні)" if abs(T) > critical_value else "Не відхиляємо H0 (змінні незалежні)"
    
    return {"rho": rho, "T": T, "critical_value": critical_value, "result": result}

def kendall_tau_correlation(X, Y):
    """
    Обчислення коефіцієнта кореляції Кендалла вручну.
    """
    n = len(X)
    concordant = 0
    discordant = 0

    # Перевіряємо всі пари (i, j)
    for i in range(n):
        for j in range(i + 1, n):
            sign_x = np.sign(X[j] - X[i])
            sign_y = np.sign(Y[j] - Y[i])

            if sign_x * sign_y > 0:
                concordant += 1  # Узгоджена пара
            elif sign_x * sign_y < 0:
                discordant += 1  # Неузгоджена пара

    # Обчислення коефіцієнта τ (tau)
    tau = (concordant - discordant) / (0.5 * n * (n - 1))

    # Обчислення статистики Z
    sigma_tau = np.sqrt((2 * (2 * n + 5)) / (9 * n * (n - 1)))
    Z = tau / sigma_tau
    critical_value = stats.norm.ppf(1 - alpha / 2)

    result = "Відхиляємо H0 (змінні залежні)" if abs(Z) > critical_value else "Не відхиляємо H0 (змінні незалежні)"
    
    return {"tau": tau, "Z": Z, "critical_value": critical_value, "result": result}


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

# Головна функція для запуску всіх тестів та виведення результатів
def main():
    # Встановлюємо seed для відтворюваності результатів
    np.random.seed(42)
    
    print("=" * 80)
    print("ЛАБОРАТОРНА РОБОТА №3: ПЕРЕВІРКА СТАТИСТИЧНИХ ГІПОТЕЗ")
    print("=" * 80)
    
    # Завдання 1: Перевірка гіпотези однорідності
    print("\n" + "=" * 80)
    print("ЗАВДАННЯ 1: ПЕРЕВІРКА ГІПОТЕЗИ ОДНОРІДНОСТІ (КРИТЕРІЙ ПУСТИХ БЛОКІВ)")
    print("=" * 80)
    
    sample_sizes = [(500, 1000), (5000, 10000), (50000, 100000)]
    for n, m in sample_sizes:
        print(f"n = {n}, m = {m}")
        result = empty_blocks_test(n, m)
        for key, value in result.items():
            print(f"{key}: {value}")
        print("-" * 50)
    
    # Завдання 2: Перевірка гіпотези незалежності
    print("\n" + "=" * 80)
    print("ЗАВДАННЯ 2: ПЕРЕВІРКА ГІПОТЕЗИ НЕЗАЛЕЖНОСТІ")
    print("=" * 80)
    
    sample_sizes = [500, 5000, 50000]  # Розміри вибірки згідно з умовою завдання
    test_cases = ['a', 'b']
    
    for case in test_cases:
        print(f"\n=== Випадок {case.upper()} ===")
        if case == 'a':
            print("Y = X * η, де η ~ U(-1,1)")
        else:
            print("Y = X + η, де η ~ U(-1,1)")
        
        for n in sample_sizes:
            print(f"\nРозмір вибірки: {n}")
            
            X, Y = generate_sample(n, case)
            
            # Візуалізація
            plt.figure(figsize=(10, 5))
            plt.scatter(X, Y, alpha=0.5)
            plt.title(f"Діаграма розсіювання (n={n}, випадок {case.upper()})")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.grid(True)
            plt.savefig(f"./imgs/scatter_{case}_{n}.png")
            plt.close()
            
            # Критерій Спірмена
            spearman_result = spearman_rank_correlation(X, Y)
            print(f"A. Критерій Спірмена:")
            print(f"   Коефіцієнт Спірмена (rho): {spearman_result['rho']:.4f}")
            print(f"   Статистика T: {spearman_result['T']:.4f}")
            print(f"   Критичне значення: {spearman_result['critical_value']:.4f}")
            print(f"   Висновок: {spearman_result['result']}")
            
            # Критерій Кендалла
            kendall_result = kendall_tau_correlation(X, Y)
            print(f"B. Критерій Кендалла:")
            print(f"   Коефіцієнт Кендалла (tau): {kendall_result['tau']:.4f}")
            print(f"   Статистика Z: {kendall_result['Z']:.4f}")
            print(f"   Критичне значення: {kendall_result['critical_value']:.4f}")
            print(f"   Висновок: {kendall_result['result']}")
    
    # Завдання 3: Перевірка гіпотези випадковості
    print("\n" + "=" * 80)
    print("ЗАВДАННЯ 3: ПЕРЕВІРКА ГІПОТЕЗИ ВИПАДКОВОСТІ (КРИТЕРІЙ ІНВЕРСІЙ)")
    print("=" * 80)
    
    a_values = [0, 0.5, 0.9]
    n = 100
    
    for a in a_values:
        print(f"\nПараметр a = {a}")
        X = generate_sample_task3(n, a)
        
        # Візуалізація послідовності
        plt.figure(figsize=(10, 5))
        plt.plot(range(n), X, 'o-', alpha=0.7)
        plt.title(f"Послідовність X при a = {a}")
        plt.xlabel('Індекс i')
        plt.ylabel('X_i')
        plt.grid(True)
        plt.savefig(f"./imgs/sequence_a{a}.png")
        plt.close()
        
        # Тест на випадковість
        result = inversion_test(X)
        print(f"Кількість інверсій (A): {result['A']}")
        print(f"Очікувана кількість інверсій (E_A): {result['E_A']:.4f}")
        print(f"Дисперсія (D_A): {result['D_A']:.4f}")
        print(f"Статистика Z: {result['Z']:.4f}")
        print(f"Критичне значення: {result['critical_value']:.4f}")
        print(f"Висновок: {result['result']}")

# Викликаємо головну функцію
if __name__ == "__main__":
    main()