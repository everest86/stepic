import matplotlib.pyplot as plt
import pandas as pd
from onnx.reference.ops.aionnxml.op_one_hot_encoder import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import (roc_auc_score, roc_curve, auc, confusion_matrix,
                             accuracy_score, classification_report,
                             precision_recall_curve, recall_score, precision_score, fbeta_score, f1_score)

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

sourceDf = pd.read_csv("salary.csv")

# целевое значение, переводим в 0 и 1, по идее так можно
targetColumn = 'salary'
sourceDf.loc[sourceDf[targetColumn] == ' >50K', targetColumn] = '1'
sourceDf.loc[sourceDf[targetColumn] == ' <=50K', targetColumn] = '0'
sourceDf[targetColumn] = pd.to_numeric(sourceDf[targetColumn])

print(sourceDf.head(2))

# # 1. Базовое решение
# print(sourceDf.isnull().sum())  # пропусков нет
#
# print(sourceDf.dtypes)  # типы данных
#
# print(sourceDf.describe())
#
# print(sourceDf[targetColumn].value_counts())  # целевые значения имеет 2 уникальных значения
#
# num_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']  # вещественные поля
#
# df = sourceDf[num_columns + [targetColumn]]
#
# x = df[num_columns]
# y = df[targetColumn]
#
# # разделение данных
# trainDf, testDf, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)
#
# # масштабирование
# scaler = StandardScaler()
# trainDf = scaler.fit_transform(trainDf)
# testDf = scaler.transform(testDf)
#
# # обучение
# model = LogisticRegression()
# model.fit(trainDf, yTrain)
#
# # получение предсказаний
# testPred = model.predict(testDf)
#
# # 0 precision = 0.83    recall =  0.94   f1 =   0.89
# # 1 precision = 0.70    recall =  0.40   f1 =   0.51
# # для ЗП <= 50 - precision (в выборку попадут только с ЗП <= 50) = 0.83%, recall (сколько % от общего кол-ва с ЗП <= 50) == 0.94 (т.е. 6% скорее не угадаем)
# # для ЗП > 50 - precision (в выборку попадут только с ЗП <= 50) = 0.70%, recall (сколько % от общего кол-ва с ЗП <= 50) == 0.40 (т.е. 60% скорее не угадаем)
# # Вывод: на ограниченном наборе данных точность предсказаний мала
# print(classification_report(yTest, testPred))
#
# # Accuracy - достаточно высокий, но он не показателен. Т.к. имеется большая разностьв количестве с зп == <=50K в 3 раза больше
# print(model.score(testDf, yTest))

# 2. Проведите первичный и визуальный анализ данных

targetColumn = 'salary'
numColumns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
catColumns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex']

# проверка workclass на уникальные значения
# вывод: найдено 9 уникальных
print(sourceDf['workclass'].value_counts().count())

# проверка education на уникальные значения
# вывод: найдено 16 уникальных
print(sourceDf['education'].value_counts().count())

# проверка marital-status на уникальные значения
# вывод: найдено 7 уникальных
print(sourceDf['marital-status'].value_counts().count())

# проверка occupation на уникальные значения
# вывод: найдено 15 уникальных
print(sourceDf['occupation'].value_counts().count())

# проверка relationship на уникальные значения
# вывод: найдено 6 уникальных
print(sourceDf['relationship'].value_counts().count())

# проверка race на уникальные значения
# вывод: найдено 5 уникальных
print(sourceDf['race'].value_counts().count())

# проверка sex на уникальные значения
# вывод: найдено 2 уникальных
print(sourceDf['sex'].value_counts().count())

# вызуальный анализ
print(sourceDf.describe())

# график зависимости salary от education
# вывод: у докторов и профи зарплата выше остальных
sourceDf[['education', targetColumn]].groupby('education')['salary'].mean().plot(kind='bar')
plt.xlabel('Education')
plt.ylabel('Average Salary')
plt.title('Average Salary by Education')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# график зависимости salary от marital-status
# вывод: у женатых военнослужащих зарплата выше остальных
sourceDf[['marital-status', targetColumn]].groupby('marital-status')['salary'].mean().plot(kind='bar')
plt.xlabel('marital-status')
plt.ylabel('Average Salary')
plt.title('Average Salary by marital-status')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# график зависимости salary от workclass
# вывод: у предпринимателя зарплата выше остальных
sourceDf[['workclass', targetColumn]].groupby('workclass')['salary'].mean().plot(kind='bar')
plt.xlabel('workclass')
plt.ylabel('Average Salary')
plt.title('Average Salary by workclass')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
