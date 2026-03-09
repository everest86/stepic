import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from DataPipeline import DataPipeline
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from EvaluatePredicts import evaluate_preds

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 1. Первичный анализ
carDf = pd.read_csv("car data.csv")
print(carDf.head(1))

# проверка на аномалии
print(carDf.describe())

# два вида трансмиссии норм
print(carDf['Transmission'].value_counts())

# два типа продавца норм
print(carDf['Seller_Type'].value_counts())

# три вида топлива норм
print(carDf['Fuel_Type'].value_counts())

# 2. Визуальный анализ

# Матрица корреляций
# Вывод: обнаружены 2 сильные корреляциии [пробег - годвыпуска] и цены [текущая - для продажи]
plt.figure(figsize=(10, 10))
corr_matrix = carDf.select_dtypes('number').corr()
corr_matrix = np.round(corr_matrix, 2)
sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap='coolwarm')
plt.show()

# зависимость текущей цены от цены продажи
# Вывод: чем выше текущая цена, тем выше цена продажи
sns.lmplot(data=carDf, x='Present_Price', y='Selling_Price')
plt.show()

# зависимость года выпуска от цены продажи
# Вывод: чем выше год выпуска, тем выше цена продажи
sns.lmplot(data=carDf, x='Year', y='Selling_Price')
plt.show()

# зависимость стоимости автомобиля от пробега
# Вывод: чем больше пробег, тем дешевле стоимость. Наблюдаются выбросы свыше 230 км
sns.jointplot(x=carDf['Kms_Driven'], y=carDf['Selling_Price'],
              kind='reg')  # sns.lmplot(x='Kms_Driven', y='Selling_Price', data=carDf)
plt.show()

# 3. Удалить категориальные признаки
carDf = carDf.select_dtypes(include=['number'])
print(carDf)

# 4. Разбить данные на обучение и тест
x = carDf.drop(columns='Selling_Price')
y = carDf['Selling_Price']
trainDf, testDf, trueTrainPrice, trueTestPrice = train_test_split(x, y, test_size=0.3, random_state=2, shuffle=True)

pipe = make_pipeline(
    DataPipeline(),
    SimpleImputer(strategy='median'),
    PolynomialFeatures(interaction_only=True),
    StandardScaler()
)

xTrain = pipe.fit_transform(trainDf)
xTest = pipe.transform(testDf)

liModel = LinearRegression()
liModel.fit(xTrain, trueTrainPrice)

xTrainPred = liModel.predict(xTrain)
xTestPred = liModel.predict(xTest)

# корреляция верных результатов с предсказаниями
# ошибка в цене может составлять до 1.032 единиц
evaluate_preds(trueTrainPrice, xTrainPred)

# корреляция верных результатов с предсказаниями
# Вывод: качество предсказания высокое
# ошибка в цене может составлять до 0.951 единиц
evaluate_preds(trueTestPrice, xTestPred)
