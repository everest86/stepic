import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.f2py.crackfortran import kindselector
from scipy import stats
import warnings

# from keplergl import KeplerGl

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
DATASET_PATH = 'housing.csv'

houses = pd.read_csv(DATASET_PATH)
print(houses.head(1))

# matplotlib график X Y
plt.figure(figsize=(16, 9))
plt.plot([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
plt.title('Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['x=y'])
plt.show()

mean = round(houses['median_house_value'].mean(), 2)
median = houses['median_house_value'].median()
mode = houses['median_house_value'].mode()[0]  # подходит вля категориальных признаков

# Гистограммы стоимости недвижимости
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
houses['median_house_value'].hist(bins=50)
plt.xlabel('median_house_value')
plt.ylabel('count')

plt.subplot(1, 2, 2)
sns.kdeplot(houses['median_house_value'], shade=True, color='blue')  # сглаженная гистограмма без столбиков
plt.xlabel('median_house_value')
plt.ylabel('count')
plt.show()

# Гистограммы стоимости недвижимости c медианой, средним и часто-встречаемым
sns.distplot(houses['median_house_value'], bins=50)  # сглаженная гистограмма со столбиками
y = np.linspace(0, 0.000005, 2)
plt.plot([mean] * 2, y, label='Mean', linestyle=':', linewidth=4)
plt.plot([median] * 2, y, label='Median', linestyle='--', linewidth=4)
plt.plot([mode] * 2, y, label='Mode', linestyle='-.', linewidth=4)  # подходит вля категориальных признаков
plt.title('Distribution of median_house_value')
plt.legend()
plt.xlabel('median_house_value')
plt.ylabel('count')
plt.show()

# Гистограммы стоимости недвижимости с медианой, средним и часто-встречаемым
houses = houses[houses['median_house_value'] < 500000]  # убираем аномалии
mean = round(houses['median_house_value'].mean(), 2)
median = houses['median_house_value'].median()
mode = houses['median_house_value'].mode()[0]  # подходит вля категориальных признаков
plt.hist(houses['median_house_value'], bins=50, density=True, alpha=0.5)  # гистограмма со столбиками
y = np.linspace(0, 0.000005, 2)
plt.plot([mean] * 2, y, label='Mean', linestyle=':', linewidth=4)
plt.plot([median] * 2, y, label='Median', linestyle='--', linewidth=4)
plt.plot([mode] * 2, y, label='Mode', linestyle='-.', linewidth=4)  # подходит вля категориальных признаков
plt.title('Distribution of median_house_value')
plt.xlabel('median_house_value')
plt.ylabel('count')
plt.legend()
plt.show()

# Анализ признаков
houses = houses[~((houses['latitude'] < -90) | (houses['latitude'] > 90))]  # Фильтр аномалий
houses = houses[~(houses['longitude'] > -50)]  # Фильтр аномалий
num_features = houses.select_dtypes(include=['float64', 'float32', 'float16'])
num_features.drop('median_house_value', axis=1, inplace=True)

# Гистограммы стоимости недвижимости по каждому столбцу
num_features.hist(figsize=(16, 16), bins=20, grid=True)
plt.show()

# Распределение по точкам. Плотность распределения зависимости параметров
grid = sns.jointplot(x=houses['median_income'], y=houses['total_bedrooms'], kind='reg')
plt.show()

grid = sns.jointplot(x=houses['total_rooms'], y=houses['total_bedrooms'], kind='reg')
plt.show()

# ящик с усами
sns.boxplot(houses['households'])  # boxplot ящик с усами сиборн
plt.show()
sns.boxplot(houses['households'],
            whis=1.5)  # boxplot ящик с усами сиборн по формуле q25-1.5(q75-q25) - нижняя граница, q75+1.5(q75-q25) - верхняя граница
plt.show()
houses['households'].plot(kind='box')  # то же самое
plt.show()

# заменяем аномалии на среднее значение
houses.loc[houses['households'] > 1000, 'households'] = houses['households'].median()
houses['households'].plot(kind='box')
plt.xlabel('Households')
plt.show()

# Расчет границ адекватности
q25 = np.quantile(houses['households'], 0.25)
q75 = np.quantile(houses['households'], 0.75)
iqr = 1.5 * (q75 - q25)
left = q25 - iqr
right = q75 + iqr
print(left, right)

# --------------------------------------------------------------------------------

# Категориальные
counts = houses['ocean_proximity'].value_counts()
sns.barplot(x=counts.index, y=counts.values)  # сиборн
counts.plot(kind='bar', title='Ocean proximity')
plt.show()

counts.plot(kind='bar', title='Ocean proximity')  # то же самое что и сиборн
plt.show()

# круговая диаграмма
plt.title('Ocean proximity')
plt.pie(counts.values, labels=counts.index, autopct='%.2f%%')  # matplotlib
plt.show()

counts.plot(kind='pie', autopct='%.2f', figsize=(16, 16))  # то же самое что matplotlib
plt.show()

# ------------------------------------------------------------------------

# Матрица корреляций для анализа взаимосвязей между признаками
plt.figure(figsize=(16, 9))
sns.set(font_scale=1.5)
corr_matrix = houses.select_dtypes('number').corr()
corr_matrix = np.round(corr_matrix, 2)
corr_matrix[np.abs(corr_matrix) < 0.3] = 0
sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap='coolwarm') # есть только в сиборне
plt.title('Correlation matrix')
plt.show()

#---------------------------------------------------------------------------

# Количественные признаки анализ
sns.jointplot(x=houses['total_bedrooms'], y=houses['median_house_value'], kind='reg') # зависимости параметров, req линейная регрессия
plt.show()

houses.plot(kind='scatter',x='total_bedrooms',y='median_house_value') # то же самое что sns.joinplot, только без линии
plt.show()

sns.jointplot(x=houses['latitude'], y=houses['median_house_value'], kind='reg') # вроде чето есть, нужно разделить
plt.show()
left_houses=houses[houses['latitude']<36] # зависимости нет
sns.jointplot(x=left_houses['latitude'], y=left_houses['median_house_value'], kind='reg')
plt.show()
right_houses=houses[houses['latitude']>=36] # прослеживается зависимость
sns.jointplot(x=right_houses['latitude'], y=right_houses['median_house_value'], kind='reg')
plt.show()

#----------------------------------------------------------------------------

# Категориальные признаки (бинарные) анализировать boxplot
plt.scatter(houses['median_house_value'], houses['ocean_proximity']) # точечный график, лучше анализировать boxplot
plt.show()

plt.figure(figsize=(30, 20))
sns.boxplot(x=houses['median_house_value'], y=houses['ocean_proximity'], width=1.5)
plt.xlabel('Median House Value')
plt.ylabel('Ocean Proximity')
plt.title('Distribution of median_house_value by Ocean Proximity')
plt.show()

# Стилизация таблиц
ocean_prox = houses.groupby('ocean_proximity').mean()[['median_house_value']]
ocean_prox = ocean_prox.sort_values('median_house_value')
ocean_prox.style.bar(align='mid') # стилизация, не отображается в PyCharm
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
