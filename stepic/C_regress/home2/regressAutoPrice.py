import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns

from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 1. Первичный анализ
carDf = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")
print(carDf.head(1))

# проверка на аномалии
print(carDf.describe())

# два вида трансмиссии норм
print(carDf['transmission'].value_counts())

# три типа продавца норм
print(carDf['seller_type'].value_counts())

# пять видов топлива норм
print(carDf['fuel'].value_counts())

# 2. Визуальный анализ
# вывод обнаружены аномалии в стоимости автомобилей 4*10^6 нужно отфильтровать
carDf.hist(bins=50)
plt.show()

# зависимость стоимости автомобиля от пробега
# Вывод: чем больше пробег, тем дешевле стоимость
sns.jointplot(x=carDf['km_driven'], y=carDf['selling_price'], kind='reg') #sns.lmplot(x='km_driven', y='selling_price', data=carDf)
plt.show()

# зависимость стоимости автомобиля от количества владельцев
# Вывод: чем больше чем больше было владельцев, тем дешевле стоимость
plt.scatter(carDf['selling_price'], carDf['owner']) # точечный график, лучше анализировать boxplot
plt.show()

sns.barplot(x=carDf['selling_price'], y=carDf['owner'])
plt.show()

sns.boxplot(x=carDf['selling_price'], y=carDf['owner'], width=1.5)
plt.show()

# Фильтруем
carDf=carDf[carDf['selling_price']<=2000000]

plt.figure(figsize=(16,8))
sns.boxplot(x=carDf['selling_price'], y=carDf['owner'], width=1.5)
plt.show()

# 3. Удалить категориальные признаки
carDf=carDf.select_dtypes(include=['number'])
print(carDf)

#4. Разбить данные на обучение и тест
trainDf, testDf, trainTruePrice, testTruePrice = train_test_split(carDf, test_size=0.3, random_state=2, shuffle=True)



