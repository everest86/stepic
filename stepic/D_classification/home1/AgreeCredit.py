import pandas as pd
import numpy as np
import warnings

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, fbeta_score)

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 1. Базовое решение

sourceLoansDf = pd.read_csv("Loan_Data.csv")

print(sourceLoansDf.head(2))

# # 1.1 удаление пустых строк
# loansDf = sourceLoansDf.dropna()
#
# # 1.2 удаление категориальных колонок
#
# x = loansDf.select_dtypes(exclude='object')
# y = loansDf['Loan_Status']
#
# trainDf, testDf, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=6, shuffle=True, stratify=y)
#
# # 1.3 обучение модели
# model = LogisticRegression()
# model.fit(trainDf, yTrain)
#
# predTrain = model.predict(trainDf)
# predTest = model.predict(testDf)
#
# # 1.4 Метрики качества
# print('TRAIN:')
# print(classification_report(yTrain, predTrain))
# print('TEST:')
# print(classification_report(yTest, predTest))

# --------------------------------------------------------------------------------------------------

# 2 первичный анализ


# x = sourceLoansDf.drop(columns=['Loan_Status', 'Loan_ID'])
# y = sourceLoansDf['Loan_Status']
#
# trainDf, testDf, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y, shuffle=True)
#
# binColumns = []
# catColumns = []
# numColumns = []
#
# # заполнение массивов с разбивкой на типы
# for column in trainDf.columns:
#     if (trainDf[column].value_counts().shape[0] == 2):
#         binColumns.append(column)
#     elif (trainDf[column].dtypes == 'object'):
#         catColumns.append(column)
#     else:
#         numColumns.append(column)
#
# # заполнение вещественных значений
# siNum = SimpleImputer(missing_values=np.nan, strategy='median')
# trainDf[numColumns] = siNum.fit_transform(trainDf[numColumns])
# testDf[numColumns] = siNum.transform(testDf[numColumns])
#
# # заполнение категориальных значений
# siCat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# trainDf[binColumns + catColumns] = siCat.fit_transform(trainDf[binColumns + catColumns])
# testDf[binColumns + catColumns] = siCat.transform(testDf[binColumns + catColumns])
#
# # проверка, что пустые ячейки заполнены
# # print(trainDf.isnull().sum())
# # print(testDf.isnull().sum())
#
# for cat_colname in catColumns + binColumns:
#     trainDf = pd.concat([trainDf, pd.get_dummies(
#         trainDf[cat_colname],
#         prefix=cat_colname,
#         dtype='int'
#     )], axis=1)
#
# trainDf.drop(columns=catColumns + binColumns, inplace=True)
#
# for cat_colname in catColumns + binColumns:
#     testDf = pd.concat([testDf, pd.get_dummies(
#         testDf[cat_colname],
#         prefix=cat_colname,
#         dtype='int'
#     )], axis=1)
#
# testDf.drop(columns=catColumns + binColumns, inplace=True)
#
# scaler = StandardScaler()
# scaler.fit(trainDf[numColumns])
#
# trainDf[numColumns] = scaler.transform(trainDf[numColumns])
# testDf[numColumns] = scaler.transform(testDf[numColumns])
#
# model = LogisticRegression()
# model.fit(trainDf, yTrain)
#
# predTrain = model.predict(trainDf)
# predTest = model.predict(testDf)
#
# print(classification_report(predTrain, yTrain))
#
# print(classification_report(predTest, yTest))
