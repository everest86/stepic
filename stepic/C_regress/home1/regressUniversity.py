import pandas as pd
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from stepic.C_regress.DataPipeline import DataPipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from EvaluatePredicts import evaluate_preds

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 1. Первичный анализ
admDf = pd.read_csv("adm_data.csv")

# Удалить ненужные поля
admDf.drop(columns='Serial No.', inplace=True)
print(admDf.head(2))

# Оценка количества пустых значений
print(admDf.isna().sum())

print(admDf.describe())

# 2. Визуальный анализ данных. Построение матрицы корреляций.
# Вывод: все параметры влияют на шансы поступления
corr_matrix = admDf.corr()
corr_matrix = np.round(corr_matrix, 2)
corr_matrix[np.abs(corr_matrix) < 0.3] = 0
sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap='coolwarm')
plt.title('Correlation matrix')
plt.show()

# Проверка на аномалии целевого значения
# Вывод: аномалий нет
admDf['Chance of Admit '].hist(bins=20)
plt.show()

# 3. Разбить данные на обучение и тест
x = admDf.drop(columns='Chance of Admit ')
y = admDf['Chance of Admit ']
trainDf, testDf, trueTrainDf, trueTestDf = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=2)

# подготовка данных
pipe = DataPipeline()
xTrainDf = pipe.fit_transform(trainDf)
xTestDf = pipe.transform(testDf)

# заполнение пропусков
si = SimpleImputer()
xTrainDfNotNull = si.fit_transform(xTrainDf)
xTestDfNotNull = si.transform(xTestDf)

# запись
xTrainDf = pd.DataFrame(xTrainDfNotNull, columns=xTrainDf.columns)
xTestDf = pd.DataFrame(xTestDfNotNull, columns=xTestDf.columns)

# масштабирование
columns = xTrainDf.columns
scaler = StandardScaler()
xTrainDf[columns] = scaler.fit_transform(xTrainDf)
xTestDf[columns] = scaler.transform(xTestDf)

print(xTestDf.head(2))

model = LinearRegression()
model.fit(xTrainDf, trueTrainDf)
print(model.coef_)  # коэффициенты колонок
print(model.intercept_)  # сдвиг
predictTrainDf = model.predict(xTrainDf)

# Вывод: модель может ошибаться на 0.063 ед.
evaluate_preds(trueTrainDf, predictTrainDf)

# проверка на тестовых данных
predictTestDf = model.predict(xTestDf)

# вывод: модель может ошибаться на 0.065 ед. на тестовых данных
evaluate_preds(trueTestDf, predictTestDf)

# ---------------------------------- Графики добавлено после работы ---------------------
sns.pairplot(admDf, corner=True)
plt.show()

sns.histplot(x="GRE Score", data=admDf, kde=True, lw=1)
plt.show()

sns.histplot(x="TOEFL Score", data=admDf, kde=True, lw=1)
plt.show()

sns.countplot(x=admDf['University Rating'])
plt.show()

sns.countplot(x='Research', data=admDf)
plt.show()

sns.countplot(x=admDf['Research'], hue=admDf['University Rating'])
plt.show()

sns.countplot(x=admDf['University Rating'], hue=admDf['Research'], color='green')
plt.show()

sns.countplot(x='SOP', data=admDf)
plt.show()

sns.countplot(x='LOR ', data=admDf)
plt.show()

# Зависимость 'Chance of Admit ' и других признаков
sns.lmplot(x='GRE Score', y='Chance of Admit ', data=admDf, hue='Research')
plt.title('Chance of Admit vs GRE score')
plt.show()

sns.lmplot(x='TOEFL Score', y='Chance of Admit ', data=admDf, hue='Research')
plt.title('Chance of Admit vs TOEFL score')
plt.show()

sns.lmplot(x='CGPA', y='Chance of Admit ', data=admDf, hue='Research')
plt.title('Chance of Admit vs CGPA')
