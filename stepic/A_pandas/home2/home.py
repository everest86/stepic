# import pandas as pd
# import numpy as np
#
# # Показать все столбцы
# pd.set_option('display.max_columns', None)
# # По желанию: увеличить ширину строки, чтобы столбцы не переносились
# pd.set_option('display.width', 1000)
#
# # Задание 1
# # Считать данные с помощью pandas
# house = pd.read_csv('kc_house_data.csv')
# # Вывести на экран первые 5 строк
# print(house.head(5))
#
# # Задание 2
# # Изучите типы данных
# print(house.info())
# # Найдите количество пропущенных ячеек в данных
# print(house.isnull().sum().sum())  # общее количество пропущенных ячеек
# # Посчитайте основные статистики по всем признакам и поизучайте их
# print(house.describe())
#
# # Задание 3
# # В каком диапазоне изменяются стоимости недвижимости?
# print(house['price'].min(), "..", house['price'].max())
# # Какую долю в среднем занимают жилая площадь от всей площади по всем домам
# totalSqftLiving = house['sqft_living'].sum()
# partOfSqftLiving = house['sqft_living'] / totalSqftLiving
# print(partOfSqftLiving) # print(house['sqft_living'].mean() * 100 / house['sqft_lot'].mean()) # правильно
# # Как много домов с разными этажами в данных?
# print(house.groupby(['floors'])['floors'].count().count())
# print(house['floors'].value_counts().count())
# # Насколько хорошее состояния у домов в данных?
# print(house.groupby(['condition'])['condition'].count())
# # Найдите года, когда построили первый дом, когда построили последний дом в данных?
# firstHouseBuiltDate = house.head(1)['date']  # дата постройки первого дома
# lastHouseBuiltDate = house.tail(1)['date']  # дата постройки первого дома
# print(np.squeeze([firstHouseBuiltDate, lastHouseBuiltDate]))
#
# # Задание 4
# # Сколько в среднем стоят дома, у которых 2 спальни?
# a = house[house['bedrooms'] == 2]['price']
# print(a.sum() / a.count())
# # Какая в среднем общая площадь домов, у которых стоимость больше 600 000?
# a = house[house['price'] > 600000]['sqft_living']
# print(a.mean())
# # Как много домов коснулся ремонт?
# a = house[(house['yr_renovated'] > 0) & (house['yr_renovated'] > house['yr_built'])]  # ремонт был
# print(a['yr_renovated'].count())
# print(a.shape[0])
# # Насколько в среднем стоимость домов с оценкой grade выше 10 отличается от стоимости домов с оценкой grade меньше 4?
# a = house[house['grade'] > 10]['price'].mean()
# b = house[house['grade'] < 4]['price'].mean()
# print(np.sqrt((a - b) ** 2))  # средняя цена отличия домов с рейтингом 10 и 4
#
# # # Задание 5
# # Выберите дом клиенту. Клиент хочет дом с видом на набережную, как минимум с тремя ванными и с подвалом. Сколько вариантов есть у клиента?
# a = house[(house['waterfront'] == 1) & (house['bathrooms'] >= 3) & (house['sqft_basement'] > 0)]['id'].count()
# print(a)
# # Выберите дом клиенту. Клиент хочет дом либо с очень красивым видом из окна, либо с видом на набережную, в очень хорошем состоянии и год постройки не меньше 1980 года. В какой ценовом диапазоне будут дома?
# a = house[((house['waterfront'] == 1) | (house['view'] == 1)) & (house['condition'] == 5) & (house['yr_built'] >= 1980)]
# print(a['price'].min(), "..", a['price'].max())
# # Выберите дом клиенту. Клиент хочет дом без подвала, с двумя этажами, стоимостью до 150000. Какая оценка по состоянию у таких домов в среднем?
# a = house[(house['sqft_basement'] == 0) & (house['floors'] == 2) & (house['price'] < 150000)]
# print(a['condition'].mean())
