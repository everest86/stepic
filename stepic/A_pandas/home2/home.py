import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 1. Считать данные с помощью pandas
house = pd.read_csv('csv/kc_house_data.csv')
# 2. Вывести на экран первые 5 строк
print(house.head(5))

# 3. Изучите типы данных. Вывод: нет пропущенных данных. Все столбцы, кроме одного - даты количественные
print(house.info())  # типы колонок
print(house.isnull().sum().sum())  # общее количество пропущенных ячеек
print(house.describe())  # основные статистики

# 4. В каком диапазоне изменяются стоимости недвижимости?
print(house['price'].min(), house['price'].max())

# 5. Какую долю в среднем занимают жилая площадь от всей площади по всем домам
print(house['sqft_living'].mean() * 100 / house['sqft_lot'].mean())

# 6. Как много домов с разными этажами в данных?
print(house['floors'].value_counts())

# 7. Насколько хорошее состояния у домов в данных?
print(house['condition'].value_counts())

# 8. Найдите года, когда построили первый дом, когда построили последний дом в данных?
print(house['yr_built'].min(), house['yr_built'].max())

# 9. Сколько в среднем стоят дома, у которых 2 спальни?
print(house[house['bedrooms'] == 2]['price'].mean())

# 10. Какая в среднем общая площадь домов, у которых стоимость больше 600 000?
print(house[house['price'] > 600000]['sqft_lot'].mean())

# 11. Как много домов коснулся ремонт?
print(house[house['yr_renovated'] > house['yr_built']].shape[0])

# 12. Насколько в среднем стоимость домов с оценкой grade выше 10 отличается от стоимости домов с оценкой grade меньше 4?
print(house[house['grade'] > 10]['price'].mean() - house[house['grade'] < 4]['price'].mean())

# 13. Выберите дом клиенту. Клиент хочет дом с видом на набережную, как минимум с тремя ванными и с подвалом. Сколько вариантов есть у клиента?
print(house[(house['waterfront'] > 0) & (house['bathrooms'] >= 3) & (house['sqft_basement'] > 0)].shape[0])

# 14. Выберите дом клиенту. Клиент хочет дом либо с очень красивым видом из окна, либо с видом на набережную, в очень хорошем состоянии и год постройки не меньше 1980 года. В какой ценовом диапазоне будут дома?
prices = house[((house['view'] == 1) | (house['waterfront'] == 1)) & (house['condition'] == 5) & (house['yr_built'] >= 1980)]['price']
print(prices.min(), prices.max())

# 15. Выберите дом клиенту. Клиент хочет дом без подвала, с двумя этажами, стоимостью до 150000. Какая оценка по состоянию у таких домов в среднем?
print(house[(house['sqft_basement'] == 0) & (house['floors'] == 2) & (house['price'] < 150000)]['condition'].mean())
