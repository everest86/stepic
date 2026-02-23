import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.f2py.crackfortran import kindselector
from scipy import stats
import warnings
import folium
# from keplergl import KeplerGl

warnings.filterwarnings('ignore')
# Показать все столбцы
pd.set_option('display.max_columns', None)
# По желанию: увеличить ширину строки, чтобы столбцы не переносились
pd.set_option('display.width', 1000)
DATASET_PATH = 'housing.csv'

houses = pd.read_csv(DATASET_PATH)
print(houses.head(1))

# plt.figure(figsize=(16, 8))
# plt.subplot(1,2,1)
# houses['median_house_value'].hist(bins=200, density=True)
# plt.xlabel('Median House Value')
# plt.ylabel('Count')
#
# plt.subplot(1,2,2)
# sns.kdeplot(houses['median_house_value'], shade=True, legend=False)
# plt.xlabel('Median House Value')
#
# plt.suptitle('Distribution of Median House Value')
# plt.show()

# # Median
# sns.distplot(houses['median_house_value'], bins=50)
# target_mean = round(houses['median_house_value'].mean(), 2)
# target_median = houses['median_house_value'].median()
# target_mode = houses['median_house_value'].mode()[0]
#
# y = np.linspace(0, 0.000005, 2)
# plt.plot([target_mean]*2, y, label='Mean',linestyle=':',linewidth=4)
# plt.plot([target_median]*2, y, label='Median',linestyle='--',linewidth=4)
# plt.plot([target_mode]*2, y, label='Mode',linestyle='-.',linewidth=4)
#
# plt.title('Distribution of median_house_value')
# plt.legend()
# plt.show()

# без аномалий
# houses=houses[houses['median_house_value']<500000]
# plt.hist(houses['median_house_value'],bins=50, density=True, alpha=0.5)
# target_mean = round(houses['median_house_value'].mean(), 2)
# target_median = houses['median_house_value'].median()
# target_mode = houses['median_house_value'].mode()[0]
#
# y = np.linspace(0, 0.000005, 10)
# plt.plot([target_mean]*2, y, label='Mean',linestyle=':',linewidth=4)
# plt.plot([target_median]*2, y, label='Median',linestyle='--',linewidth=4)
# plt.plot([target_mode]*2, y, label='Mode',linestyle='-.',linewidth=4)
# plt.title('Distribution of median_house_value')
# plt.legend()
# plt.show()

# Анализ признаков
houses = houses[~((houses['latitude'] < -90) | (houses['latitude'] > 90))]  # Фильтр аномалий
houses = houses[~(houses['longitude'] > -50)]  # Фильтр аномалий

num_features = houses.select_dtypes(include=['float64', 'float32', 'float16'])
num_features.drop('median_house_value', axis=1, inplace=True)
# гистограммы по каждому столбцу
num_features.hist(figsize=(16, 16), bins=20, grid=True)
# точки
# grid=sns.jointplot(x=houses['median_income'], y=houses['total_bedrooms'], kind='reg')
# grid=sns.jointplot(x=houses['total_rooms'], y=houses['total_bedrooms'], kind='reg')

# ящик с усами
# sns.boxplot(houses['households'])
# houses['households'].plot(kind='box')
# sns.boxplot(houses['households'], whis=1.5)
# houses.loc[houses['households'] > 1000, 'households'] = houses['households'].median()
# houses['households'].plot(kind='box')
# plt.xlabel('Households')
# q25 = np.quantile(houses['households'], 0.25)
# q50 = np.quantile(houses['households'], 0.5)
# q75 = np.quantile(houses['households'], 0.75)
# iqr = 1.5 * (q75 - q25)
# left = q25 - iqr
# right = q75 + iqr
# print(left, q25, q50, q75, right)


# Категориальные
a = houses['ocean_proximity'].value_counts()
# sns.barplot(x=a.index, y=a.values)
# a.plot(kind='bar', title='Ocean proximity')

# круговая диаграмма
# plt.title('Ocean proximity')
# plt.pie(a.values, labels=a.index, autopct='%.2f%%')
# a.plot(kind='pie', autopct='%.2f', figsize=(16, 16))

# Матрица корреляций для анализа взаимосвязей между признаками
# grid=sns.jointplot(x=houses['total_rooms'], y=houses['total_bedrooms'], kind='reg')
# sns.set(font_scale=1.5)
# corr_matrix=houses.select_dtypes('number').corr()
# corr_matrix = np.round(corr_matrix,2)
# corr_matrix[np.abs(corr_matrix)<0.3]=0
# sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap='coolwarm')
# plt.title('Correlation matrix')


# Количественные признаки
# sns.jointplot(x=houses['total_bedrooms'], y=houses['median_house_value'], kind='reg')
# houses.plot(kind='scatter',x='total_bedrooms',y='median_house_value') # нет линейной зависимости
# sns.jointplot(x=houses['latitude'], y=houses['median_house_value'], kind='reg') # вроде чето есть, нужно разделить
# left_houses=houses[houses['latitude']<36] # зависимости нет
# sns.jointplot(x=left_houses['latitude'], y=left_houses['median_house_value'], kind='reg')
# right_houses=houses[houses['latitude']>=36] # прослеживается зависимость
# sns.jointplot(x=right_houses['latitude'], y=right_houses['median_house_value'], kind='reg')

# # Категориальные признаки (бинарные) анализировать boxplot
# plt.scatter(houses['median_house_value'], houses['ocean_proximity'])
# sns.boxplot(x=houses['median_house_value'], y=houses['ocean_proximity'], width=1.5)
# plt.xlabel('Median House Value')
# plt.ylabel('Ocean Proximity')
# plt.title('Distribution of median_house_value by Ocean Proximity')

# Стилизация таблиц
# ocean_prox = houses.groupby('ocean_proximity').mean()[['median_house_value']]
# ocean_prox = ocean_prox.sort_values('median_house_value')
# ocean_prox.style.bar(align='mid') # стилизация, не отображается в PyCharm
#
plt.show()

# Геоданные folium

# this_map = folium.Map(prefer_canvas=True)
# def plotDot(point, color):
#
#     folium.CircleMarker(
#         location=[point.latitude, point.longitude],
#         radius=2,
#         weight=5,
#         color=color,
#         popup=point.median_house_value
#     ).add_to(this_map)
# houses.iloc[:2000].apply(plotDot, axis=1, color='#3388FF')
# this_map.fit_bounds(this_map.get_bounds())
# this_map.save(outfile='./geo_folium.html')
#
#
# # Геоданные Kepler
#
# map_ = KeplerGl(height=700)
# map_.add_data(houses, 'Data')
# map_.save_to_html(file_name='./geo_folium.html')

# выводы:
# 1. строим матрицу корреляций для обзора зависимостей
# для количественных и когда целевое тоже - использовать sns.joinplot, d.plot(kind='scatter',..) - линейный график, гистограмма, точечный, ящик с усами
# для категориальных (целевой, обычный) - sns.boxplot - столбики, пирогова диаграмма