import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math

# Встановлюємо рівень значимості
alpha = 0.05

# Функція для генерації вибірки з показникового розподілу
def generate_exponential_sample(lam, size):
    """Генерує вибірку з показникового розподілу з параметром lam"""
    return np.random.exponential(scale=1/lam, size=size)


# Функція для перетворення показникового розподілу в рівномірний
def transform_to_uniform(x, lam):
    """Перетворює показниковий розподіл в рівномірний за формулою F(x)=1-e^(-lambda*x)"""
    return 1 - np.exp(-lam * x)


# Функція для генерації вибірки з рівномірного розподілу
def generate_uniform_sample(size):
    """Генерує вибірку з рівномірного розподілу на [0,1]"""
    return np.random.uniform(0, 1, size=size)


# Розміри вибірок для тестування
sample_sizes = [1000, 10000, 100000]

# Параметри розподілів
lambda_true = 1
lambda_false = 1.2

# Завдання 1: Критерій Колмогорова
def kolmogorov_test(sample, distribution_func, params=None):
    """
    Перевірка гіпотези за допомогою критерію Колмогорова
    sample - вибірка
    distribution_func - функція розподілу
    params - параметри розподілу
    """
    n = len(sample)
    sample_sorted = np.sort(sample)

    # Обчислюємо емпіричну функцію розподілу
    ecdf = np.arange(1, n + 1) / n

    # Обчислюємо теоретичну функцію розподілу
    if params is not None:
        tcdf = distribution_func(sample_sorted, *params)
    else:
        tcdf = distribution_func(sample_sorted)

    # Знаходимо максимальну різницю
    D_n = np.max(np.abs(ecdf - tcdf))

    # Критичне значення для рівня значимості alpha
    critical_value = stats.kstwo.ppf(1 - alpha, n)

    # Перевіряємо гіпотезу
    reject = D_n > critical_value

    return {
        'D_n': D_n,
        'critical_value': critical_value,
        'reject': reject
    }

# Завдання 2: Критерій хі-квадрат
def chi_square_test(sample, distribution_func, params=None, num_bins=None):
    """
    Перевірка гіпотези за допомогою критерію хі-квадрат
    sample - вибірка
    distribution_func - функція розподілу
    params - параметри розподілу
    num_bins - кількість інтервалів
    """
    n = len(sample)

    # Визначаємо кількість інтервалів за формулою
    if num_bins is None:
        num_bins = int(1.72 * n**(1/3))

    # Обчислюємо межі інтервалів
    bins = np.linspace(0, 1, num_bins + 1)

    # Обчислюємо кількість елементів у кожному інтервалі
    observed, _ = np.histogram(sample, bins=bins)

    # Обчислюємо очікувану кількість елементів у кожному інтервалі
    expected = np.ones(num_bins) * n / num_bins

    # Обчислюємо статистику хі-квадрат
    chi2 = np.sum((observed - expected) ** 2 / expected)

    # Ступені свободи
    df = num_bins - 1

    # Критичне значення для рівня значимості alpha
    critical_value = stats.chi2.ppf(1 - alpha, df)

    # Перевіряємо гіпотезу
    reject = chi2 > critical_value

    return {
        'chi2': chi2,
        'critical_value': critical_value,
        'reject': reject,
        'num_bins': num_bins
    }

# Виправлена функція empty_boxes_test
def empty_boxes_test(sample, num_bins=None):
    """
    Перевірка гіпотези за допомогою критерію пустих ящиків
    sample - вибірка
    num_bins - кількість інтервалів
    """
    n = len(sample)

    # Визначаємо кількість інтервалів за формулою
    if num_bins is None:
        # Використовуємо формулу ro = 2, R = n/ro як у прикладі
        ro = 2
        num_bins = int(n / ro)

    # Обчислюємо межі інтервалів
    bins = np.linspace(0, 1, num_bins + 1)

    # Обчислюємо кількість елементів у кожному інтервалі
    observed, _ = np.histogram(sample, bins=bins)

    # Кількість пустих ящиків
    empty_boxes = np.sum(observed == 0)

    # Очікувана кількість пустих ящиків для рівномірного розподілу
    # Використовуємо формулу e^(-ro) як у прикладі
    ro = n / num_bins  # Середня кількість елементів у ящику
    e_pow_m_ro = math.exp(-ro)
    expected_empty = num_bins * e_pow_m_ro

    # Дисперсія кількості пустих ящиків
    variance = num_bins * e_pow_m_ro * (1 - (1 + ro) * e_pow_m_ro)
    
    # Обчислюємо Z-статистику
    z_stat = (empty_boxes - expected_empty) / math.sqrt(variance) if variance > 0 else 0

    # Критичне значення для рівня значимості alpha
    z_gamma = stats.norm.ppf(1 - alpha/2)  # Для двостороннього тесту

    # Критичне значення для кількості пустих ящиків
    critical_value = expected_empty + z_gamma * math.sqrt(variance)

    # Перевіряємо гіпотезу
    reject = abs(z_stat) > z_gamma

    return {
        'empty_boxes': empty_boxes,
        'expected_empty': expected_empty,
        'z_stat': z_stat,  # Змінено ім'я ключа з 'z' на 'z_stat'
        'critical_value': critical_value,
        'z_gamma': z_gamma,
        'reject': reject,
        'num_bins': num_bins
    }

# Виправлена функція Smirnov test
def smirnov_test(sample1, sample2):
    """
    Перевірка гіпотези однорідності за допомогою критерію Смирнова
    sample1, sample2 - дві вибірки
    """
    # Використовуємо функцію з scipy для обчислення статистики
    result = stats.ks_2samp(sample1, sample2)
    
    n = len(sample1)
    m = len(sample2)
    
    # Обчислюємо емпіричні функції розподілу
    def ecdf(x, sample):
        return np.sum(sample <= x) / len(sample)
    
    # Об'єднуємо вибірки для обчислення всіх можливих значень
    combined = np.concatenate((sample1, sample2))
    combined_sorted = np.sort(combined)
    
    # Обчислюємо різницю між емпіричними функціями розподілу
    differences = np.array([abs(ecdf(x, sample1) - ecdf(x, sample2)) for x in combined_sorted])
    
    # Знаходимо максимальну різницю
    D_mn = np.max(differences)
    
    # Критичне значення для рівня значимості alpha
    critical_value = np.sqrt(-0.5 * np.log(alpha/2) * (1/n + 1/m))
    
    # Перевіряємо гіпотезу
    reject = result.pvalue < alpha
    
    return {
        'D': D_mn,
        'critical_value': critical_value,
        'p_value': result.pvalue,
        'reject': reject
    }

# Виконання всіх завдань для різних розмірів вибірок
for n in sample_sizes:
    print(f"\n{'='*50}")
    print(f"Розмір вибірки: {n}")
    print(f"{'='*50}")

    # Генеруємо вибірки
    sample_true = generate_exponential_sample(lambda_true, n)
    sample_false = generate_exponential_sample(lambda_false, n)

    # Перетворюємо в рівномірний розподіл
    uniform_true = transform_to_uniform(sample_true, lambda_true)
    uniform_false = transform_to_uniform(sample_true, lambda_false)

    # Завдання 1: Критерій Колмогорова
    print("\n### Завдання 1: Критерій Колмогорова")

    # a) H0: F(x) = 1-e^(-2x), коли насправді F(x) = 1-e^(-2x)
    result_1a = kolmogorov_test(uniform_true, lambda x: x)
    print(f"a) H0: F(x) = 1-e^(-{lambda_true}x), коли насправді F(x) = 1-e^(-{lambda_true}x)")
    print(f"   D_n = {result_1a['D_n']:.4f}, критичне значення = {result_1a['critical_value']:.4f}")
    print(f"   Висновок: {'відхиляємо H0' if result_1a['reject'] else 'не відхиляємо H0'}")

    # b) H0: F(x) = 1-e^(-3x), коли насправді F(x) = 1-e^(-2x)
    result_1b = kolmogorov_test(uniform_false, lambda x: x)
    print(f"b) H0: F(x) = 1-e^(-{lambda_false}x), коли насправді F(x) = 1-e^(-{lambda_true}x)")
    print(f"   D_n = {result_1b['D_n']:.4f}, критичне значення = {result_1b['critical_value']:.4f}")
    print(f"   Висновок: {'відхиляємо H0' if result_1b['reject'] else 'не відхиляємо H0'}")

    # Завдання 2: Критерій хі-квадрат
    print("\n### Завдання 2: Критерій хі-квадрат")

    # Визначаємо кількість інтервалів
    num_bins = int(1.72 * n**(1/3))

    # a) H0: F(x) = 1-e^(-2x), коли насправді F(x) = 1-e^(-2x)
    result_2a = chi_square_test(uniform_true, None, num_bins=num_bins)
    print(f"a) H0: F(x) = 1-e^(-{lambda_true}x), коли насправді F(x) = 1-e^(-{lambda_true}x)")
    print(f"   χ² = {result_2a['chi2']:.4f}, критичне значення = {result_2a['critical_value']:.4f}, кількість інтервалів = {result_2a['num_bins']}")
    print(f"   Висновок: {'відхиляємо H0' if result_2a['reject'] else 'не відхиляємо H0'}")

    # b) H0: F(x) = 1-e^(-3x), коли насправді F(x) = 1-e^(-2x)
    result_2b = chi_square_test(uniform_false, None, num_bins=num_bins)
    print(f"b) H0: F(x) = 1-e^(-{lambda_false}x), коли насправді F(x) = 1-e^(-{lambda_true}x)")
    print(f"   χ² = {result_2b['chi2']:.4f}, критичне значення = {result_2b['critical_value']:.4f}, кількість інтервалів = {result_2b['num_bins']}")
    print(f"   Висновок: {'відхиляємо H0' if result_2b['reject'] else 'не відхиляємо H0'}")

    # Завдання 3: Критерій пустих ящиків
    print("\n### Завдання 3: Критерій пустих ящиків")

    # a) H0: F(x) = 1-e^(-2x), коли насправді F(x) = 1-e^(-2x)
    result_3a = empty_boxes_test(uniform_true)
    print(f"a) H0: F(x) = 1-e^(-{lambda_true}x), коли насправді F(x) = 1-e^(-{lambda_true}x)")
    print(f"   Кількість пустих ящиків = {result_3a['empty_boxes']}, очікувана = {result_3a['expected_empty']:.2f}")
    print(f"   Z = {result_3a['z_stat']:.4f}, критичне значення = {result_3a['critical_value']:.4f}, кількість інтервалів = {result_3a['num_bins']}")
    print(f"   Висновок: {'відхиляємо H0' if result_3a['reject'] else 'не відхиляємо H0'}")

    # b) H0: F(x) = 1-e^(-3x), коли насправді F(x) = 1-e^(-2x)
    result_3b = empty_boxes_test(uniform_false)
    print(f"b) H0: F(x) = 1-e^(-{lambda_false}x), коли насправді F(x) = 1-e^(-{lambda_true}x)")
    print(f"   Кількість пустих ящиків = {result_3b['empty_boxes']}, очікувана = {result_3b['expected_empty']:.2f}")
    print(f"   Z = {result_3a['z_stat']:.4f}, критичне значення = {result_3a['critical_value']:.4f}, кількість інтервалів = {result_3a['num_bins']}")
    print(f"   Висновок: {'відхиляємо H0' if result_3b['reject'] else 'не відхиляємо H0'}")

    # Завдання 4: Критерій однорідності Смирнова
    print("\n### Завдання 4: Критерій однорідності Смирнова")

    # Генеруємо дві вибірки з однаковим розподілом
    sample1 = generate_exponential_sample(lambda_true, n)
    sample2 = generate_exponential_sample(lambda_true, n)

    # Генеруємо дві вибірки з різними розподілами
    sample3 = generate_exponential_sample(lambda_true, n)
    sample4 = generate_exponential_sample(lambda_false, n)

    # a) H0: F1(x) = F2(x), коли насправді F1(x) = F2(x)
    result_4a = smirnov_test(sample1, sample2)
    print(f"a) H0: F1(x) = F2(x), коли насправді F1(x) = F2(x)")
    print(f"   D = {result_4a['D']:.4f}, p-значення = {result_4a['p_value']:.4f}")
    print(f"   Висновок: {'відхиляємо H0' if result_4a['reject'] else 'не відхиляємо H0'}")

    # b) H0: F1(x) = F2(x), коли насправді F1(x) ≠ F2(x)
    result_4b = smirnov_test(sample3, sample4)
    print(f"b) H0: F1(x) = F2(x), коли насправді F1(x) ≠ F2(x)")
    print(f"   D = {result_4b['D']:.4f}, p-значення = {result_4b['p_value']:.4f}")
    print(f"   Висновок: {'відхиляємо H0' if result_4b['reject'] else 'не відхиляємо H0'}")

# Візуалізація результатів для n = 100
n = 100
sample_true = generate_exponential_sample(lambda_true, n)
uniform_true = transform_to_uniform(sample_true, lambda_true)
uniform_false = transform_to_uniform(sample_true, lambda_false)

plt.figure(figsize=(15, 10))

# Гістограма для показникового розподілу
plt.subplot(2, 2, 1)
plt.hist(sample_true, bins=20, density=True, alpha=0.7, label='Вибірка')
x = np.linspace(0, max(sample_true), 1000)
plt.plot(x, lambda_true * np.exp(-lambda_true * x), 'r-', label=f'f(x) = {lambda_true}e^(-{lambda_true}x)')
plt.title('Показниковий розподіл')
plt.legend()

# Гістограма для рівномірного розподілу (правильне перетворення)
plt.subplot(2, 2, 2)
plt.hist(uniform_true, bins=20, density=True, alpha=0.7, label='Перетворена вибірка')
plt.plot([0, 1], [1, 1], 'r-', label='f(x) = 1')
plt.title('Рівномірний розподіл (правильне перетворення)')
plt.legend()

# Гістограма для рівномірного розподілу (неправильне перетворення)
plt.subplot(2, 2, 3)
plt.hist(uniform_false, bins=20, density=True, alpha=0.7, label='Перетворена вибірка')
plt.plot([0, 1], [1, 1], 'r-', label='f(x) = 1')
plt.title('Рівномірний розподіл (неправильне перетворення)')
plt.legend()

# Емпірична функція розподілу
plt.subplot(2, 2, 4)
sorted_data = np.sort(uniform_true)
ecdf = np.arange(1, n + 1) / n
plt.step(sorted_data, ecdf, label='Емпірична ФР')
plt.plot([0, 1], [0, 1], 'r-', label='Теоретична ФР')
plt.title('Емпірична функція розподілу')
plt.legend()

plt.tight_layout()
plt.savefig('results_visualization.png')
plt.show()

print("\nВізуалізація збережена у файлі 'results_visualization.png'")