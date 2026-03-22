import pandas as pd
import numpy as np
import warnings

from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, auc, confusion_matrix,
                             accuracy_score, classification_report,
                             precision_recall_curve, recall_score, precision_score, fbeta_score, f1_score)

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

sourceDf = pd.read_csv("creditcard.csv")

print(sourceDf.head())

# пропущенных нет
print(sourceDf.describe())
# типы все number
print(sourceDf.dtypes)

# 1. Базовое решение
x = sourceDf.drop(columns=['Class'])
y = sourceDf['Class']

trainDf, testDf, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y, shuffle=True)

# Масштабирование
scaler = StandardScaler()
trainDf=scaler.fit_transform(trainDf)
testDf=scaler.transform(testDf)

# Обучение
model=LogisticRegression()
model.fit(trainDf, yTrain)
print(model.coef_)

predTrain=model.predict(trainDf)
predTest=model.predict(testDf)

print(classification_report(yTrain, predTrain))
print(classification_report(yTest, predTest))

# 1.3. Выберете и посчитайте метрику качества
# Вывод: для тестовых данных предсказание что это мошенник f1-score=0.73