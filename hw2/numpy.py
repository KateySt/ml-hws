# -*- coding: utf-8 -*-
"""numpy.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1globPsilzf2dOYbRZeOM8bQOvpvESeED

Для успішного виконання домашки використовуйте документацію [NumPy](https://numpy.org/doc/stable/).

Зверніть увагу на такі методи `numpy`:
* *arange*
* *zeros*
* *ones*
* *full*
* *nonzero*
* *eye*
* *random*
* *sort*
* *sum*
* *min*
* *mean*
"""

import numpy as np

# Завдання 1: Створити масив розміром 3x3 зі значеннями від 0 до 8
# Hints: np.arange
arr1 = np.arange(0, 9).reshape(3, 3)

# Завдання 2: Створити вектор розміром 10, заповнений нулями
arr2 = np.zeros(10)

# Завдання 3: Створити вектор розміром 10, заповнений одиницями
arr3 = np.ones(10)

# Завдання 4: Створити вектор розміром 10, заповнений значеннями 5
# Hints: np.full
arr4 = np.full(10, 5)

# Завдання 5: Створити вектор з числами від 10 до 49
arr5 = np.arange(10, 50)

# Завдання 6: Реверсувати порядок елементів вектора з завдання 5
arr6 = np.arange(10, 50)[::-1]

# Завдання 7: Створити 9x9 матрицю зі значеннями від 0 до 80
# Hints: np.arange
arr7 = np.arange(0, 81).reshape(9, 9)

# Завдання 8: Знайти індекси ненульових елементів вектора [1,2,0,0,4,0]
# Hints: np.nonzero
arr8 = np.nonzero(np.array([1, 2, 0, 0, 4, 0]))

# Завдання 9: Створити одиничну матрицю розміром 3x3
# Hints: np.eye
arr9 = np.eye(3)

# Завдання 10: Створити випадковий вектор розміром 10 та відсортувати його
arr10 = np.random.rand(10)
arr10_sorted = np.sort(arr10)

# Завдання 11: Створити матрицю 5x5 з випадковими цілими числами в діапазоні від 0 до 10
arr11 = np.random.randint(0, 11, size=(5, 5))

# Завдання 12: Обчислити суму всіх елементів матриці з завдання 11
arr11_sum = np.sum(arr11)

# Завдання 13: Знайти найменше значення в кожному рядку матриці з завдання 11
arr11_min_row = np.min(arr11, axis=1)

# Завдання 14: Обчислити середнє значення кожного стовпця матриці з завдання 11
arr11_mean_col = np.mean(arr11, axis=0)

# Завдання 15: Створити випадковий вектор розміром 15 та замінити максимальне значення на -1
# Hints: np.argmax
arr15 = np.random.randint(1, 101, size=15)
arr15[arr15 == np.max(arr15)] = -1

# Завдання 16: Перетворити випадковий вектор розміром 10 в матрицю розміром 2x5
arr16 = np.random.randint(1, 101, size=10).reshape(2, 5)

# Завдання 17: Знайти середнє значення елементів матриці з завдання 16
arr16_mean = np.mean(arr16)