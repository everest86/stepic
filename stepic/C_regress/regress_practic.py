import numpy as np
import pandas as pd
import pickle  # сохранение модели

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from matplotlib.pyplot import scatter
# Разделение датасета - для обучения и тестирования
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

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

# 2. удаляем ненужные поля
x = df.drop(columns='median_house_value')  # матрица без целевого значения
y = df['median_house_value']  # целевые значения

# 5. Разбиение данных. Параметры: обучение, тест, обучение с целевыми значениями, тест с целевыми значениями. test_size - доля данных которая идет на тест,
# random_state - фиксация состояния
x_train_orig, x_test_orig, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=1)

# 6. Подготовка
pipeline = DataPipeline()
pipeline.fit(x_train_orig)
x_train = pipeline.transform(x_train_orig)
x_test = pipeline.transform(x_test_orig)  # на тестовой выборке (не вызывать метод fit())

# 7. Пропуски
# print(x_train.isnull().sum())
si = SimpleImputer(strategy='median')
si.fit(x_train)
x_train_np = si.transform(x_train)  # numpy array
x_train = pd.DataFrame(x_train_np, columns=x_train.columns)
x_test_np = si.transform(x_test)
x_test = pd.DataFrame(x_test_np, columns=x_test.columns)

# 8. Масштабирование признаков
feature_names_for_stand = x_train.columns

standartScaler = StandardScaler()
x_train_standart_scaler = standartScaler.fit_transform(x_train[feature_names_for_stand])  # передача признаков
x_test_standart_scaler = standartScaler.transform(x_test[feature_names_for_stand])  # масштабирование тестовой выборки

x_train[feature_names_for_stand] = x_train_standart_scaler  # для признаков делать перезапись
x_test[feature_names_for_stand] = x_test_standart_scaler

# 9. Сохранение выборок
x_train.to_csv(PREPARED_DATASET_PATH_TRAIN, index=False, sep=";")
x_test.to_csv(PREPARED_DATASET_PATH_TEST, index=False, sep=";")


# 10. построение моделей. Метрики

def evaluate_preds(true_values, pred_values, save=False):
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


lr_model = LinearRegression()
# обучение модели
lr_model.fit(x_train, y_train)
# print(lr_model.coef_)  # коэффициенты
# print(lr_model.intercept_)  # сдвиг
y_train_pred = lr_model.predict(x_train)  # предсказания
y_train_pred_manual = np.sum(
    lr_model.coef_ * x_train.iloc[0].values) + lr_model.intercept_  # предсказания то же самое только вручную подсчет
# print(y_train_pred[0]) # то же самое print(y_train_pred_manual)

# print(y_train_pred_manual)
# постпроцессинг
y_train_pred = np.clip(y_train_pred, a_min=10000, a_max=500000)  # обрезать
# визуализация
# evaluate_preds(y_train, y_train_pred)

# предсказания на тесте
# вывод: метрики схожи, модель обучена нормально
y_test_pred = lr_model.predict(x_test)
y_test_pred = np.clip(y_test_pred, a_min=10000, a_max=500000)
# evaluate_preds(y_test, y_test_pred)

# 11. Улучшение модели
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(
    DataPipeline(),
    SimpleImputer(strategy='median'),
    StandardScaler()
)
# print(pipe)
pipe.fit(x_train_orig, y_train)  # обучение на исходных
x_train = pipe.transform(x_train_orig)
x_test = pipe.transform(x_test_orig)
# print(x_train)

model = LinearRegression()
model.fit(x_train, y_train)
y_train_pred = model.predict(x_train)
# Постпроцессинг
y_train_pred = np.clip(y_train_pred, a_min=10000, a_max=500000)
evaluate_preds(y_train, y_train_pred)

model.fit(x_test, y_test)
y_test_pred = model.predict(x_test)
# Постпроцессинг
y_test_pred = np.clip(y_test_pred, a_min=10000, a_max=500000)
# evaluate_preds(y_test, y_test_pred)

# Полиноминальные признаки. degree=3 - степень
poly = PolynomialFeatures(interaction_only=True, degree=3)
# print(poly.fit_transform(x_train_orig.iloc[:1])) # переумножение признаков, каждый с каждым переумножается и больше точек получается

scaler = StandardScaler()
pipe = make_pipeline(
    DataPipeline(),
    SimpleImputer(strategy='median'),
    PolynomialFeatures(interaction_only=True),
    scaler
)
x_train = pipe.fit_transform(x_train_orig)
x_test = pipe.transform(x_test_orig)

model = LinearRegression()
model.fit(x_train, y_train)
y_train_pred = model.predict(x_train)
# Постпроцессинг
y_train_pred = np.clip(y_train_pred, a_min=10000, a_max=500000)
evaluate_preds(y_train, y_train_pred)  # вывод: R стало больше

model.fit(x_test, y_test)
y_test_pred = model.predict(x_test)
# Постпроцессинг
y_test_pred = np.clip(y_test_pred, a_min=10000, a_max=500000)
# evaluate_preds(y_test, y_test_pred)

with open(SCALER_FILE_PATH, 'wb') as filename:
    pickle.dump(scaler, filename)

with open(MODEL_FILE_PATH, 'wb') as filename:
    pickle.dump(model, filename)

with open(MODEL_FILE_PATH, 'rb') as filename:
    my_model = pickle.load(filename)

print(my_model)  # чтение записанной модели

# 12. Версии sklearn должны быть одинаковы
# pip freeze >> requrements.txt
