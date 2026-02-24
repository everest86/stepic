import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sympy import rotations

# Показать все столбцы
pd.set_option('display.max_columns', None)
# По желанию: увеличить ширину строки, чтобы столбцы не переносились
pd.set_option('display.width', 1000)
wines = pd.read_csv('wine/winequality-red.csv', sep=';')

# print(wines.describe())
#
# print(wines.info())


wines['good'] = (wines['quality'] > 5).astype(int)
print(wines.head(1))

# # 1. Матрица корреляций
# plt.figure(figsize = (16,8))
# corr_matrix=wines.corr()
# corr_matrix = np.round(corr_matrix,2)
# corr_matrix[np.abs(corr_matrix)<0.3]=0
# sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap='coolwarm')
# plt.show()

# # 2. Зависимость качества от летучей кислотности
# a=wines.groupby('quality')['volatile acidity'].median().reset_index().sort_values('quality')
# print(a)
# plt.figure(figsize = (10,7))
# plt.plot(a['volatile acidity'], a['quality'])
# plt.title('Падение качества при превышении volatile acidity')
# plt.xlabel('volatile acidity')
# plt.ylabel('quality')
# plt.show()

# # 3. Зависимость качества от лимонной кислоты
# a=wines.groupby('quality')['citric acid'].median().reset_index()
# print(a)
# plt.figure(figsize=(16,8))
# plt.bar(a['quality'], a['citric acid'])
# plt.title('Влияние citric acid на качество вина')
# plt.xlabel('quality')
# plt.ylabel('citric acid')
# plt.show()

# # 4. Две зависимости от качества
# # вывод: в хороших винах меньше volatile_acidity и больше citric acid
# plt.figure(figsize=(10, 7))
# acidity = wines.groupby('quality')['volatile acidity'].median().reset_index().sort_values('quality')
# citric = wines.groupby('quality')['citric acid'].median().reset_index()
# w = 0.4
# offset = 0.2
# plt.bar(acidity['quality'] - offset, acidity['volatile acidity'], width=w, label='volatile_acidity')
# plt.bar(citric['quality'] + offset, citric['citric acid'], width=w, label='citric_acid')
# plt.xlabel('Качество')
# plt.ylabel('Показатели')
# plt.legend(['volatile acidity', 'citric acid'])
# plt.show()

# # 5. Количество хороших и плохих вин  (круговая диаграмма)
# a=wines['good'].value_counts().reset_index() # или так a=wines['good'].value_counts()
# print(a)
# plt.figure(figsize=(10,7))
# plt.pie(a['count'], labels=a['good'], autopct='%.2f%%') # и так plt.pie(a, autopct='%.2f%%')
# plt.title('Доля хороших вин')
# plt.legend(['плохие', 'хорошие'])
# plt.show()

# # 6. зависимость качества вина от содержания спирта (ящик с усами boxplot)
# # вывод: хорошие отличаются от плохих по медианным показателям признака 'alcohol' (меньше в плохих винах)
# alco1=wines[wines['good']==1]['alcohol']
# alco0=wines[wines['good']==0]['alcohol']
# plt.figure(figsize=(10, 7))
# plt.boxplot([alco0, alco1], showfliers=False)
# plt.title('Уровень крепости вина')
# plt.xlabel('Качество')
# plt.ylabel('Alcohol')
# plt.xticks([1,2],['Плохое', 'Хорошее'])
# plt.show()

# # 6. зависимость качества вина от volatile acidity (ящик с усами boxplot)
# # вывод: хорошие отличаются от плохих по медианным показателям признака 'volatile acidity' (меньше в хороших винах)
# alco1=wines[wines['good']==1]['volatile acidity']
# alco0=wines[wines['good']==0]['volatile acidity']
# plt.figure(figsize=(10, 7))
# plt.boxplot([alco0, alco1], showfliers=False)
# plt.title('Уровень volatile acidity')
# plt.xlabel('Качество')
# plt.ylabel('volatile acidity')
# plt.xticks([1,2],['Плохое', 'Хорошее'])
# plt.show()

# # 7. Относительные показатели
# # вывод: 'alcohol' меньше в плохих, 'sulfur dioxide' больше в плохих и 'total sulfur dioxide' больше в плохих
# groupped_median = wines.groupby('good').median().drop('quality', axis=1).transpose().reset_index()
# groupped_median = groupped_median.rename(columns={'index': 'feature', 0: 'bad', 1: 'good'})
# groupped_median.columns.name = None
# groupped_median['median_diff'] = groupped_median['bad'] - groupped_median['good']
# print(groupped_median)
#
# plt.figure(figsize=(10, 6))
# # количество делений
# n_ticks = np.arange(len(groupped_median))
# plt.bar(n_ticks, groupped_median['median_diff'])
# plt.title('Median difference between wines')
# plt.xlabel('Признаки')
# plt.ylabel('Разница медианы')
# plt.xticks(n_ticks, groupped_median['feature'], rotation=30)
# plt.grid(True)
# plt.show()
