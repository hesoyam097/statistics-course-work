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
    
    # Генеруємо вибірки X ~ U(0,1) та Y ~ U(0,2)
    n1 = 100
    n2 = 100
    X = np.random.uniform(0, 1, n1)
    Y = np.random.uniform(0, 2, n2)
    
    # Параметри для тестів
    test_params = [
        {"m": 10, "k": 5},
        {"m": 20, "k": 10},
        {"m": 50, "k": 2}
    ]
    
    for i, params in enumerate(test_params):
        m, k = params["m"], params["k"]
        print(f"\nТест {i+1}: m = {m}, k = {k}")
        
        result = empty_blocks_test(X, Y, m, k)
        
        print(f"Кількість пустих блоків (mu): {result['mu']:.4f}")
        print(f"Очікувана кількість пустих блоків (E_mu): {result['E_mu']:.4f}")
        print(f"Дисперсія (D_mu): {result['D_mu']:.4f}")
        print(f"Статистика U: {result['U']:.4f}")
        print(f"Критичне значення: {result['critical_value']:.4f}")
        print(f"Висновок: {result['result']}")
    
    # Завдання 2: Перевірка гіпотези незалежності
    print("\n" + "=" * 80)
    print("ЗАВДАННЯ 2: ПЕРЕВІРКА ГІПОТЕЗИ НЕЗАЛЕЖНОСТІ")
    print("=" * 80)
    
    sample_sizes = [10, 50, 100]
    test_cases = ['a', 'b']
    
    for case in test_cases:
        print(f"\nВипадок {case.upper()}:")
        if case == 'a':
            print("Y = X + eps, де eps ~ U(0, 0.2)")
        else:
            print("Y = X^2 + eps, де eps ~ U(0, 0.2)")
        
        for n in sample_sizes:
            print(f"\nРозмір вибірки: {n}")
            
            X, Y = generate_dependent_sample(n, case)
            
            # Візуалізація даних
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.scatter(X, Y, alpha=0.7)
            plt.title(f"Діаграма розсіювання (n={n})")
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.hist2d(X, Y, bins=20, cmap='Blues')
            plt.title(f"Двовимірна гістограма (n={n})")
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.colorbar(label='Частота')
            
            plt.tight_layout()
            plt.savefig(f"./imgs/scatterplot_{case}_{n}.png")
            plt.close()
            
            print("A. Критерій Спірмена:")
            spearman_result = spearman_test(X, Y)
            print(f"   Коефіцієнт кореляції Спірмена (rho): {spearman_result['rho']:.4f}")
            print(f"   Статистика T: {spearman_result['T']:.4f}")
            print(f"   Критичне значення: {spearman_result['critical_value']:.4f}")
            print(f"   p-значення: {spearman_result['p_value']:.4f}")
            print(f"   Висновок: {spearman_result['result']}")
            
            print("\nB. Критерій Кендалла:")
            kendall_result = kendall_test(X, Y)
            print(f"   Коефіцієнт кореляції Кендалла (tau): {kendall_result['tau']:.4f}")
            print(f"   Статистика Z: {kendall_result['Z']:.4f}")
            print(f"   Критичне значення: {kendall_result['critical_value']:.4f}")
            print(f"   p-значення: {kendall_result['p_value']:.4f}")
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