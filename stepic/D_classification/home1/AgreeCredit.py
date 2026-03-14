import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, fbeta_score)

from stepic.D_classification.home1.DataPipeline import DataPipeline

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
#
# # --------------------------------------------------------------------------------------------------
#
# # 2 первичный анализ
#
# # количество строк, столбцов
# print(sourceLoansDf.shape)
#
# # типы данных
# print(sourceLoansDf.dtypes)
#
# # сгруппированные значения
# print(sourceLoansDf.describe())
#
# # --------------------------------------------------------------------------------------------------
#
# # 3 визуальный анализ данных
#
# # матрица корреляции
# # вывод: на целевой показатель в основном влияет кредитная история,
# # а на размер кредита - доход и в меньшей степени - доп.доход
# sourceLoansDf['statusInt'] = 0
# sourceLoansDf.loc[sourceLoansDf['Loan_Status'] == 'Y', 'statusInt'] = 1
#
# plt.figure(figsize=(16, 16))
# matrixCorr = sourceLoansDf.select_dtypes(exclude='object').corr()
# corr_matrix = np.round(matrixCorr, 2)
# corr_matrix[np.abs(corr_matrix) < 0.01] = 0
# sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap='coolwarm')
# plt.show()
#
# # график зависимости размера кредита от дохода заявителя
# # Вывод: наблюдается слабенькая линейная зависимость
# sns.jointplot(sourceLoansDf, x='ApplicantIncome', y='LoanAmount', kind='reg')
# plt.show()
#
# # какой гендер берет больше кредитов
# # вывод М берет больше кредитов, чем Ж
# a = sourceLoansDf['Gender'].value_counts()
# sns.barplot(x=a.index, y=a.values)
# plt.show()
#
# # у кого больше в среднем доход М или Ж
# # Вывод: у М общий доход в среднем выше чем у Ж
# a = sourceLoansDf[['Gender', 'ApplicantIncome', 'CoapplicantIncome']].groupby(['Gender']).mean()
# sns.barplot(x=a.index, y=[a.values[0].sum(), a.values[1].sum()])
# plt.show()
#
# # кому в рамках текущей выборке чаще одобряют кредит
# a = sourceLoansDf[['Gender', 'statusInt']].groupby(['Gender']).sum()
# print(a)
# plt.pie(a['statusInt'], labels=a.index, autopct='%.2f%%')
# plt.show()

# --------------------------------------------------------------------------------------------------

# 4. Разбить данные на обучение и тест
x = sourceLoansDf.drop(columns=['Loan_Status', 'Loan_ID'])
y = sourceLoansDf['Loan_Status']

trainDf, testDf, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y, shuffle=True)

# 5. Сделайте предобработку данных с помощью класса и пайплайна

pipe = DataPipeline()
trainDf = pipe.fit_transform(trainDf)

# заполнение вещественных значений
siNum = SimpleImputer(missing_values=np.nan, strategy='median')
trainDf[pipe.numColumns] = siNum.fit_transform(trainDf[pipe.numColumns])
testDf[pipe.numColumns] = siNum.transform(testDf[pipe.numColumns])

# заполнение категориальных значений
siCat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
trainDf[pipe.binCatColumns] = siCat.fit_transform(trainDf[pipe.binCatColumns])
testDf[pipe.binCatColumns] = siCat.transform(testDf[pipe.binCatColumns])

# преобразование категориальных признаков
for cat_colname in pipe.binCatColumns:
    trainDf = pd.concat([trainDf, pd.get_dummies(
        trainDf[cat_colname],
        prefix=cat_colname,
        dtype='int'
    )], axis=1)

trainDf.drop(columns=pipe.binCatColumns, inplace=True)

for cat_colname in pipe.binCatColumns:
    testDf = pd.concat([testDf, pd.get_dummies(
        testDf[cat_colname],
        prefix=cat_colname,
        dtype='int'
    )], axis=1)

testDf.drop(columns=pipe.binCatColumns, inplace=True)

# масштабирование данных
scaler = StandardScaler()
scaler.fit(trainDf[pipe.numColumns])

trainDf[pipe.numColumns] = scaler.transform(trainDf[pipe.numColumns])
testDf[pipe.numColumns] = scaler.transform(testDf[pipe.numColumns])

# 6 обучение модели
model = LogisticRegression()
model.fit(trainDf, yTrain)

predTrain = model.predict(trainDf)
predTest = model.predict(testDf)

print(classification_report(predTrain, yTrain))

print(classification_report(predTest, yTest))
