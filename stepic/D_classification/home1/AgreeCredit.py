import pandas as pd
import warnings

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix)

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 1. Базовое решение

# загрузка данных
sourceLoanDf = pd.read_csv("Loan_Data.csv").head()

# часть данных
print(sourceLoanDf)

# пустые данные в каждой колонке
print(sourceLoanDf.isnull().sum())

# добавление числового целевого поля
sourceLoanDf['status'] = 0
sourceLoanDf.loc[sourceLoanDf['Loan_Status'] == 'Y', 'status'] = 1

# типы данных
print(sourceLoanDf.dtypes)

# удаление пустых данных
loanDf = sourceLoanDf[~sourceLoanDf['LoanAmount'].isnull()]
print(loanDf.isnull().sum())

# удаление категориальных данных
loanDf = loanDf.select_dtypes(exclude='object')
print(loanDf.head())

x = loanDf.drop(columns=['status'])
y = loanDf['status']

print(y)

# разделение
trainDf, testDf, yTrainDf, yTestDf = train_test_split(x, y,
                                                      test_size=0.2, random_state=3, shuffle=True)

# масштабироваие
scaler = StandardScaler()
scaler.fit(trainDf)
trainDf = scaler.transform(trainDf)
testDf = scaler.transform(testDf)

print(testDf)

# обучение модели
model = LogisticRegression()
model.fit(trainDf, yTrainDf)

predTrain = model.predict(trainDf)
predTest = model.predict(testDf)

# Метрика Accuracy для тенировочных данных
trainAccuracy = accuracy_score(yTrainDf, predTrain)
# Метрика Accuracy для тенировочных данных
testAccuracy = accuracy_score(yTestDf, predTest)
print("Train Accuracy:", trainAccuracy, "Test Accuracy:", testAccuracy)
# Вывод: на тренировочных данных Accuracy=1, на тестовых = 0

# PR - кривая
print(confusion_matrix(yTrainDf, predTrain))