# -*- coding: utf-8 -*-
"""pandas.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rP8VOR1l3n3XWpc6B3E3AvQt7-C7VKUD

[Pandas docs](https://pandas.pydata.org)
"""

import pandas as pd

# Завдання 1: Створити серію зі списку чисел [10, 20, 30, 40, 50]
ser1 = pd.Series([10, 20, 30, 40, 50])

# Завдання 2: Створити DataFrame зі списку списків [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
df2 = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Завдання 3: Задати індекси для серії з завдання 1 рядками "a", "b", "c", "d", "e"
# Hints: series.index
ser1 = pd.Series([10, 20, 30, 40, 50])
ser1.index = ['a', 'b', 'c', 'd', 'e']

# Завдання 4: Перейменувати стовпці DataFrame з завдання 2 на "A", "B", "C"
# Hints: df.columns
df2 = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df2.columns = ['A', 'B', 'C']

# Завдання 5: Знайти максимальний елемент в серії з завдання 1
ser1_max = ser1.max()

# Завдання 6: Знайти середнє значення для кожного стовпця DataFrame з завдання 2
df2_mean = df2.mean()

# Завдання 7: Знайти суму елементів в серії з завдання 1
ser1_sum = ser1.sum()

# Завдання 8: Вибрати всі рядки DataFrame з завдання 2, де значення стовпця "A" менше 5
df2_filtered = df2[df2['A'] < 5]

# Завдання 9: Додати новий стовпець "D" до DataFrame з завдання 2 зі значеннями [10, 20, 30]
df2['D'] = [10, 20, 30]

# Завдання 10: Видалити стовпець "B" з DataFrame з завдання 2
df2 = df2.drop(columns=['B'])

