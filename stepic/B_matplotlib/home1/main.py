import pandas as pd
from scipy.stats import planck
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sympy.physics.units import percent

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
employees = pd.read_csv('employer/Employee.csv')
print(employees.head(1))

# 1. Исследовать данные с применением не менее 5 диаграмм из урока
print(employees.info())

# 1.1. Гистограммы по количественным столбцам
employees.hist(figsize=(16, 16), bins=50, grid=True)
plt.show()

# 1.2. Круговая диаграмма по уровню зарплаты
# Вывод: высокооплачиваемых сотрудников больше в 3 раза остальных
plt.figure(figsize=(16, 16))
employees['PaymentTier'].value_counts().plot(kind='pie', autopct='%.2f', figsize=(16, 16))
plt.title('Круговая диаграмма количества на каждом уровне оплаты труда')
plt.show()

# 1.3. Круговая диаграмма по уровню зарплаты
# Вывод: высокооплачиваемых сотрудников больше в 3 раза остальных
plt.figure(figsize=(16, 16))
employees['PaymentTier'].value_counts().plot(kind='pie', autopct='%.2f', figsize=(16, 16))
plt.title('Круговая диаграмма количества на каждом уровне оплаты труда')
plt.show()

# 1.4. Оставшиеся и уволившиеся от года присоединения к компании
# Вывод: с 2013 по 2014 в основном нанимали Bachelors и PHD
plt.figure(figsize=(10, 9))
sns.boxplot(x=employees['JoiningYear'], y=employees['Education'], width=1.5)
plt.show()

# 1.5. Кто больше в среднем заработывает М или Ж
# Вывод: с
employees['gender_int']=0
employees.loc[employees['Gender']=='Male', 'gender_int']=1
a=employees.groupby('gender_int')['PaymentTier'].mean()
print(a.values())
plt.figure(figsize=(10, 9))
plt.pie(a.values(), labels=a.index, autopct='%.2f%%')
plt.show()

# # 1. Матрица корреляций
# # вывод: сильных зависимостей нет
# corr_matrix = employees.select_dtypes('number').corr()
# corr_matrix = np.round(corr_matrix, 2)
# corr_matrix[np.abs(corr_matrix) < 0.1] = 0
# sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap='coolwarm')
# plt.title('Correlation matrix')
# plt.show()
#
# # 2. График зависимости количества сотрудников сгруппированных по уровню образования
# a = employees['Education'].value_counts()
# plt.subplot(1, 3, 1)
# sns.barplot(x=a.index, y=a.values)  # вывод: бакалавров больше
# # plt.pie(a.values, labels=a.index, autopct='%.2f%%')
#
# # 3. График зависимости количества сотрудников сгруппированных по городу
# a = employees['City'].value_counts()
# plt.subplot(1, 3, 2)
# sns.barplot(x=a.index, y=a.values)  # вывод: родом из Bangalore сотрудников больше
# # plt.pie(a.values, labels=a.index, autopct='%.2f%%')
#
# # 4. График количества мужчин и женщин
# a = employees['Gender'].value_counts()
# plt.subplot(1, 3, 3)
# sns.barplot(x=a.index, y=a.values)  # вывод: мужчин больше чем женщин
# # plt.pie(a.values, labels=a.index, autopct='%.2f%%')
# plt.show()
#
# # 5. График зависимости оплаты от образования. Вывод: уровень оплаты труда не зависит от образования
# plt.figure(figsize=(16, 8))
# plt.scatter(employees['PaymentTier'], employees['Education'])
# plt.xlabel('Payment Tier')
# plt.ylabel('Education')
# plt.title('Distribution of PaymentTier by Education')
# plt.show()
#
# # 6. График зависимости оплаты от образования.
# # Вывод: возраст сотрудников для PaymentTier in (2,3) схожий.
# plt.figure(figsize=(16, 8))
# sns.boxplot(x=employees['PaymentTier'], y=employees['Age'], whis=1.5)
# plt.xlabel('Payment Tier')
# plt.ylabel('Age')
# plt.title('Distribution of PaymentTier by Age')
# plt.show()
