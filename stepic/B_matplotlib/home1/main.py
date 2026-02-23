import pandas as pd
from scipy.stats import planck
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sympy.physics.units import percent

# Показать все столбцы
pd.set_option('display.max_columns', None)
# По желанию: увеличить ширину строки, чтобы столбцы не переносились
pd.set_option('display.width', 1000)
employees = pd.read_csv('employer/Employee.csv')
print(employees.head(1))

# 0. Подготовка
print(employees.info())
num_features = employees.select_dtypes(include=['int64'])
# проверка на аномалии
num_features.hist(figsize=(16, 16), bins=20, grid=True)
plt.show()

# 1. Матрица корреляций
# вывод: сильных зависимостей нет
corr_matrix = employees.select_dtypes('number').corr()
corr_matrix = np.round(corr_matrix, 2)
corr_matrix[np.abs(corr_matrix) < 0.1] = 0
sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap='coolwarm')
plt.title('Correlation matrix')
plt.show()

# 2. График зависимости количества сотрудников сгруппированных по уровню образования
a = employees['Education'].value_counts()
plt.subplot(1, 3, 1)
sns.barplot(x=a.index, y=a.values)  # вывод: бакалавров больше
# plt.pie(a.values, labels=a.index, autopct='%.2f%%')

# 3. График зависимости количества сотрудников сгруппированных по городу
a = employees['City'].value_counts()
plt.subplot(1, 3, 2)
sns.barplot(x=a.index, y=a.values)  # вывод: родом из Bangalore сотрудников больше
# plt.pie(a.values, labels=a.index, autopct='%.2f%%')

# 4. График количества мужчин и женщин
a = employees['Gender'].value_counts()
plt.subplot(1, 3, 3)
sns.barplot(x=a.index, y=a.values)  # вывод: мужчин больше чем женщин
# plt.pie(a.values, labels=a.index, autopct='%.2f%%')
plt.show()

# 5. График зависимости оплаты от образования. Вывод: уровень оплаты труда не зависит от образования
plt.figure(figsize=(16, 8))
plt.scatter(employees['PaymentTier'], employees['Education'])
plt.xlabel('Payment Tier')
plt.ylabel('Education')
plt.title('Distribution of PaymentTier by Education')
plt.show()

# 6. График зависимости оплаты от образования.
# Вывод: возраст сотрудников для PaymentTier in (2,3) схожий.
plt.figure(figsize=(16, 8))
sns.boxplot(x=employees['PaymentTier'], y=employees['Age'], whis=1.5)
plt.xlabel('Payment Tier')
plt.ylabel('Age')
plt.title('Distribution of PaymentTier by Age')
plt.show()
