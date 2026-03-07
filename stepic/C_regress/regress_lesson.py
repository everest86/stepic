import numpy as np
import pandas as pd
import pickle  # сохранение модели

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Разделение датасета - для обучения и тестирования
from sklearn.model_selection import train_test_split

# Модели
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# заполнитель
from sklearn.impute import SimpleImputer

# Метрики качества
from sklearn.metrics import mean_squared_error as mse, r2_score as r2
from sympy.physics.paulialgebra import evaluate_pauli_product

from stepic.C_regress.DataPipeline import DataPipeline

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# исхлдные жанные
DATASET_PATH = 'housing.csv'
# для обучения
PREPARED_DATASET_PATH_TRAIN = 'housing_train.csv'
# для тестов
PREPARED_DATASET_PATH_TEST = 'housing_test.csv'
# модель для масштабирования данных
SCALER_FILE_PATH = 'scaler.pkl'
# модель предсказывает стоимость недвижимости
MODEL_FILE_PATH = 'model.pkl'

# 1. получение данных
df = pd.read_csv(DATASET_PATH, sep=',')
# print(df.head().head(1))

# 2. типы данных - один object (нужно что-то с ней сделать)
# print(df.dtypes)

# 3. Визуальный анализ данных (убрать выбросы)
df['median_house_value'].hist(bins=50)
# plt.show()
# убираем выбросы
df = df[df['median_house_value'] <= 500000]
df['median_house_value'].hist(bins=50)
# plt.show()
# print(df.shape)

# 4. удаляем ненужные поля
x = df.drop(columns='median_house_value')  # матрица без целевого значения
y = df['median_house_value']  # целевые значения

# 5. Разбиение данных. Параметры: обучение, тест, обучение с целевыми значениями, тест с целевыми значениями. test_size - доля данных которая идет на тест,
# random_state - фиксация состояния
x_train_orig, x_test_orig, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=1)
# print(x_train_orig.index)

# 6. Подготовка
pipeline = DataPipeline()
pipeline.fit(x_train_orig)
x_train = pipeline.transform(x_train_orig)
x_test = pipeline.transform(x_test_orig)  # на тестовой выборке (не вызывать метод fit())

# print(x_train.head(1))

# 7. Пропуски
# print(x_train.isnull().sum())
si = SimpleImputer(strategy='median')
si.fit(x_train)
# print(si.statistics_) # массив медиан, тоже самое print(x_train.median())
# заменяем пропуски
x_train_np = si.transform(x_train)  # numpy array
# print(x_train_np) # матрица null заменены на медианное значение
# переводим в датафрейм
x_train = pd.DataFrame(x_train_np, columns=x_train.columns)
# print(x_train.head(10))

# 8. Масштабирование признаков
feature_names_for_stand = x_train.columns

# MinMaxScaler лучше использовать для моделей которые работают с расстояниями (x[i]-min)/(max-min)
# minMaxScaler = MinMaxScaler()
# x_train_min_max_scaler = minMaxScaler.fit_transform(x_train)
# x_test_min_max_scaler = minMaxScaler.transform(x_test)  # масштабирование тестовой выборки
# x_train = pd.DataFrame(x_train_min_max_scaler, columns=feature_names_for_stand)
# # print(x_train.head(10))

# StandardScaler лучше использовать для линейных моделей  (x[i]-mean)/std
standartScaler = StandardScaler()
x_train_standart_scaler = standartScaler.fit_transform(x_train[feature_names_for_stand])  # передача признаков
x_test_standart_scaler = standartScaler.transform(x_test[feature_names_for_stand])  # масштабирование тестовой выборки

x_train[feature_names_for_stand] = x_train_standart_scaler  # для признаков делать перезапись
x_test[feature_names_for_stand] = x_test_standart_scaler

# print(standartScaler.mean_, standartScaler.var_)
# print(x_train_standart_scaler)

# 9. Сохранение выборок
x_train.to_csv(PREPARED_DATASET_PATH_TRAIN, index=False, sep=";")
x_test.to_csv(PREPARED_DATASET_PATH_TEST, index=False, sep=";")

# 10. построение моделей. Метрики
# MSE = 1/n * sum(y-pow(y[пред],2)) - под капотом, идеальное значение=0
y      = np.array([100, 200, 100])
y_pred = np.array([105, 190, 80])

# RMSE = sqrt(1/n * sum(y-pow(y[пред],2))) - кол-во единиц на сколько ошибается модель, идеальное значение=0
print(np.sqrt(np.mean((y - y_pred) ** 2)))
# pow(R,2) = 1- (1/n * sum(y-pow(y[пред],2))) / 1/n * sum(y-pow(y[mean],2)), идеальное значение=1
y      = np.array([100, 200, 100])
y_pred = np.array([1050, 1900, 800])
y_mean = np.array([133, 133, 133])
print(1-(np.mean((y - y_pred) ** 2)/np.mean((y - y_mean) ** 2)))

def evaluate_preds(true_values, pred_values, save=False):
    """Оценка качества модели и график preds vs true"""

    print("R2:\t" + str(round(r2(true_values, pred_values), 3)) + "\n" +
          "RMSE:\t" + str(round(np.sqrt(mse(true_values, pred_values)), 3)) + "\n" +
          "MSE:\t" + str(round(mse(true_values, pred_values), 3))
          )

    plt.figure(figsize=(8, 8))

    sns.scatterplot(x=pred_values, y=true_values)
    plt.plot([0, 500000], [0, 500000], linestyle='--', color='black')  # диагональ, где true_values = pred_values

    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    plt.title('True vs Predicted values')

    if save == True:
        plt.savefig('report.png')
    plt.show()