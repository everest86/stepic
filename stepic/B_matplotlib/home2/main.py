import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 1. загрузка данных
houses = pd.read_csv('kc_house_data.csv')
print(houses.head(1))

# 2. Гистограмма цены, выше 3.5 млн.ед. наблюдаются аномалии
# вывод: большая часть домов продается 100 тыщ до 1 млр
plt.figure(figsize=(16, 8))
# убираем аномалии
houses = houses[houses['price'] < 3.5 * pow(10, 6)]
sns.distplot(houses['price'], bins=100)
y = np.linspace(0, 0.000005, 2)
target_median = np.median(houses['price'])
target_mean = round(houses['price'].mean(), 2)
target_mode = houses['price'].mode()[0]
plt.plot([target_mean] * 2, y, label='Mean', linestyle=':', linewidth=4)
plt.plot([target_median] * 2, y, label='Median', linestyle='--', linewidth=4)
plt.plot([target_mode] * 2, y, label='Mode', linestyle='-.', linewidth=4)
plt.xlabel('price')
plt.ylabel('count')
plt.title('Distribution of price')
plt.show()

# 3. Распрделение квадратуры живой площади
# вывод: большее количество квартир с площадью от 700 до 6000 кв.метров.
plt.figure(figsize=(16, 8))
# убираем аномалии
houses = houses[houses['sqft_living'] < 8000]
sns.distplot(houses['sqft_living'], bins=100)
target_median = np.median(houses['sqft_living'])
target_mean = round(houses['sqft_living'].mean(), 2)
target_mode = houses['sqft_living'].mode()[0]
y = np.linspace(0, 0.0006, 2)
plt.plot([target_mean] * 2, y, label='Mean', linestyle=':', linewidth=4)
plt.plot([target_median] * 2, y, label='Median', linestyle='--', linewidth=4)
plt.plot([target_mode] * 2, y, label='Mode', linestyle='-.', linewidth=4)
plt.xlabel('sqft_living')
plt.ylabel('count')
plt.title('Distribution of sqft_living')

print(houses['sqft_living'].value_counts())
plt.show()

# # 3. Распрделение года постройки
# # вывод: в долгосрочной перспективе квартир с каждым годом строится больше, аномалий не выявлено
plt.figure(figsize=(16, 8))
# убираем аномалии
sns.distplot(houses['yr_built'], bins=20)
target_median = np.median(houses['yr_built'])
target_mean = round(houses['yr_built'].mean(), 2)
target_mode = houses['yr_built'].mode()[0]
y = np.linspace(0, 0.0006, 2)
plt.plot([target_mean] * 2, y, label='Mean', linestyle=':', linewidth=4)
plt.plot([target_median] * 2, y, label='Median', linestyle='--', linewidth=4)
plt.plot([target_mode] * 2, y, label='Mode', linestyle='-.', linewidth=4)
plt.xlabel('yr_built')
plt.ylabel('count')
plt.title('Distribution of yr_built')
print(houses['yr_built'].value_counts())
plt.show()

# # 4. Распределение домов от наличия видов на набережную
houses.loc[houses['view'] > 0, 'view'] = 1
a = houses.groupby('view').count()['id']
print(a)
plt.figure(figsize=(10, 8))
plt.pie(a, autopct='%.2f%%')
plt.title('Доля домов с видом на набережную')
plt.legend(['Остальные', 'С видом на набережную'])
plt.show()
