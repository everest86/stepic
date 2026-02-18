import pandas as pd
import numpy as np
from scipy import stats

DATASET_PATH = './housing.csv'
PREPARED_DATASET_PATH = './housing_prepared.csv'
# Показать все столбцы
pd.set_option('display.max_columns', None)
# По желанию: увеличить ширину строки, чтобы столбцы не переносились
pd.set_option('display.width', 1000)
df = pd.read_csv(DATASET_PATH, sep=',')

# # all rows
# print(df)
#
# # Первые 2
# print(df.head(2))
# # Последние 2
# print(df.tail(2))
#
# # random 5 files
# print(df.sample(5))
#
# # random all rows
# print(df.sample(frac=1))
#
# # random 50% rows
# print(df.sample(frac=0.5))

# # rows and columns count
# print(df.shape)

# # first and second indexes, and step size
# print(df.index)

# # custom columns
# print(df[['total_rooms', 'total_bedrooms']])
# print(df['total_rooms'])
# print(df.total_rooms)

# # вывести строки по условию
# print(df.total_rooms==880) == print(df['total_rooms']==880)
# print(df.population > 400)
# print(df['population'] > 400)
# print(df[(df['population'] > 20)].head(10))
# print(df[((df['population'] > 20) & (df['households']>200))].head(10))
# print(df[((df['population'] > 20) | (df['households']>200))].head(10))
# print(~(df['population'] > 400)) # отридцание
# print(df[~(df['population'] > 20)].head(10))

# index
# tmp=df[~(df['population']>100)]
# print(tmp.loc[34]) # поиск по индексу
# print(tmp.iloc[0]) # поиск по порядку элемента в списке
# print(tmp.loc[55:61]) # поиск по срезу индекса
# print(tmp.iloc[:5]) # возвращает первые 5 элементов
# print(tmp.loc[[34,61], ['latitude','longitude']]) # поиск по срезу индекса с нужными колонками
# print(tmp.iloc[:5, [0,1]]) # возвращает первые 5 элементов

# # add new data
# df['NEAR_BAY'] = 0  # new column
# df.loc[df['ocean_proximity'] == 'NEAR BAY', 'NEAR_BAY'] = 1
# print(df)
#
# # drop
# df.drop(columns=['NEAR_BAY'], inplace=True) # удаление по значению
# print(df.head())

# типы столбцов
# print(df.dtypes)
# print(df['ocean_proximity'].dtype)
# print(df['id']) # столбик
# print(df['id'].values) # np-array
# print(type(df['id'].values))
# df['id']=df['id'].astype(str)
# print(df['id'].dtype)

# # количественные переменные
# df_num_features = df.select_dt ypes(include=['float64', 'int64']) # возвращает только количественные данные
# print(df_num_features.head())

# # информация о данных
# print(df.describe())
#
# print(df.info()) # информация о метаданных

a = np.array([1, 2, 2, 3, 4, 5, 5])
# mean=a.sum()/len(a)
# # print(mean) # среднее значение
#
# s = a-mean # отклонение от среднего
# variance=np.mean(s**2) # дисперсия
# std=np.sqrt(variance) # срендеквадратичное отклонение
# print(variance, std)
#
# print(np.mean(a)) # срендее
# print(np.var(a), np.std(a)) # срендеквадратичное отклонение

# print(np.median(a)) # середина массива
#
# print(np.quantile(a, 0.5)) # квантиль 0,5 соответствует середина массива
# print(np.quantile(a, 0.25)) # квантиль

# print(stats.mode(a)) # выводит частовстречающееся значение и сколько раз
print(pd.Series(a).mode()) # выводит частовстречающееся значение и сколько раз (работает лучше, чем предыдущий вариант)

print(df['total_rooms'].value_counts()) # количество каждого из значений
print(df['total_rooms'].mode()) # выводит частовстречающееся значение и сколько раз (работает лучше, чем предыдущий вариант)
print(df[df['total_rooms']==1527].shape) # количество строк равное значению 1527
