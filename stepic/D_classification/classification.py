import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (roc_auc_score, roc_curve, auc, confusion_matrix, \
                             accuracy_score, classification_report, \
                             precision_recall_curve, recall_score)
from sklearn.linear_model import LogisticRegression

import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

DATASET_PATH = 'employee.csv'
PREP_DATASET_TRAIN_PATH = 'employee_prep_train.csv'
PREP_DATASET_TEST_PATH = 'employee_prep_test.csv'

employeesDf = pd.read_csv(DATASET_PATH)
print(employeesDf.head(5))

print(employeesDf.shape)

print(employeesDf.info())

# обзор целевой характеристики
print(employeesDf['left'].value_counts())

print(employeesDf.describe())

# Обзор номинативных признаков
for cat_colname in employeesDf.select_dtypes(include='object').columns:
    print(str(cat_colname) + '\n\n' + str(employeesDf[cat_colname].value_counts()) + '\n' + '*' * 100 + '\n')

# разбиение данных
x = employeesDf.drop(columns="left")
y = employeesDf['left']

trainDf, testDf, trueTrainDf, trueTestDf = train_test_split(x, y, test_size=0.2, random_state=2, shuffle=True,
                                                            stratify=y)

print(trainDf.isnull().sum())

# отделить типы
num_features = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']
bin_features = ['Work_accident', 'promotion_last_5years']
cat_features = ['department', 'salary']

from sklearn.impute import SimpleImputer

si_med = SimpleImputer(strategy='median')
si_med.fit(trainDf[num_features])  # обучение тренировочных данных

print(si_med.statistics_)

trainDf[num_features] = si_med.transform(trainDf[num_features])
testDf[num_features] = si_med.transform(testDf[num_features])

# заполнение категориальных
si_mode = SimpleImputer(strategy='most_frequent')
si_mode.fit(trainDf[cat_features + bin_features])

trainDf[cat_features + bin_features] = si_mode.transform(trainDf[cat_features + bin_features])
testDf[cat_features + bin_features] = si_mode.transform(testDf[cat_features + bin_features])

# проверка на пропуски
print(trainDf.isnull().sum())

# # Ошибка конвертации: ValueError: could not convert string to float: 'sales'
# # lr = LogisticRegression()
# # lr.fit(trainDf, trueTrainDf)

from sklearn.preprocessing import OneHotEncoder

# перевод в числовой формат
print(pd.get_dummies(trainDf[cat_colname], dtype='int'))

for cat_colname in cat_features:
    trainDf = pd.concat([trainDf, pd.get_dummies(
        trainDf[cat_colname],
        prefix=cat_colname,
        dtype='int'  # UPD. перевод в int, а не bool
    )
                         ], axis=1)

print(trainDf.head())
# Удаление категориальных признаков
trainDf.drop(columns=cat_features, inplace=True)
print(trainDf.head())

print(trainDf.shape, testDf.shape)

# тоже самое для тестовых данных
for cat_colname in cat_features:
    testDf = pd.concat([testDf, pd.get_dummies(
        testDf[cat_colname],
        prefix=cat_colname,
        dtype='int'  # UPD. перевод в int, а не bool
    )
                        ], axis=1)

# print(trainDf.head())
# Удаление категориальных признаков
testDf.drop(columns=cat_features, inplace=True)
print(testDf.head())

print(testDf.shape, testDf.shape)

# Масштабирование
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(trainDf[trainDf.columns])

trainDf[trainDf.columns] = scaler.transform(trainDf[trainDf.columns])
testDf[testDf.columns] = scaler.transform(testDf[testDf.columns])

pd.concat([trainDf, trueTrainDf], axis=1).to_csv(PREP_DATASET_TRAIN_PATH, index=False)
pd.concat([testDf, trueTestDf], axis=1).to_csv(PREP_DATASET_TEST_PATH, index=False)

# обучение модели
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(trainDf, trueTrainDf)

print(testDf)

pred_train = lr.predict(trainDf)
pred_test = lr.predict(testDf)

pred_proba_test = lr.predict_proba(testDf)
print(pred_proba_test[:5])

print(pred_test)

# Подсчет метрик Accuracy
accuracy_train = accuracy_score(trueTrainDf, pred_train)
accuracy_test = accuracy_score(trueTestDf, pred_test)
print(f'Accuracy на трейне {accuracy_train:.2f}')
print(f'Accuracy на тесте {accuracy_test:.2f}')

# строим матрицу
print(confusion_matrix(trueTestDf, pred_test))
print(pd.crosstab(trueTestDf, pred_test))

from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay.from_estimator(lr, testDf, trueTestDf, cmap=plt.cm.Blues)
plt.show()

print(classification_report(trueTestDf, pred_test))

# Micro, Macro, Weighted

from sklearn.metrics import f1_score, precision_score, recall_score, \
    classification_report, confusion_matrix, accuracy_score, f1_score

true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
pred = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0]

print(confusion_matrix(true, pred))
# micro
precision = 9 / 15
print(precision, accuracy_score(true, pred))

recall = 9 / 15
print(recall, accuracy_score(true, pred))

# f1 score
print(2 * (precision * recall) / (precision + recall))
print(f1_score(true, pred, average='micro'))

# MACRO
precision_0 = 4 / (4 + 5)
precision_1 = 5 / (5 + 1)
macro_pr = (precision_1 + precision_0) / 2
print(macro_pr)
print(precision_score(true, pred, average='macro'))

# recall
recall_1 = 5 / (5 + 5)
recall_0 = 4 / (4 + 1)
macro_rec = (recall_1 + recall_0) / 2
print(macro_rec)
print(recall_score(true, pred, average='macro'))

# f score
f_score_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0)
f_score_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1)
f_score = (f_score_0 + f_score_1) / 2
print(f_score)
print(f1_score(true, pred, average='macro'))

# WEIGHTED
# precision
zero = true.count(0)
one = true.count(1)
size = len(true)
print(zero / size * precision_0 + one / size * precision_1)
print(precision_score(true, pred, average='weighted'))

# recall
print(zero / size * recall_0 + one / size * recall_1)
print(recall_score(true, pred, average='weighted'))

# F мера
f_score_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0)
f_score_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1)
print(zero / size * f_score_0 + one / size * f_score_1)
print(f1_score(true, pred, average='weighted'))

# графики
pd.Series(pred_proba_test[:, 1]).hist()
plt.show()

import numpy as np

pred_test = np.where(pred_proba_test[:, 1] <= 0.2, 1, 0)
print(classification_report(trueTestDf, pred_test))

# UPD 2025-07-29
# plot_precision_recall_curve больше нет в sklearn
# plot_precision_recall_curve(lr, X_test, y_test);

# можно пользоваться PrecisionRecallDisplay
from sklearn.metrics import PrecisionRecallDisplay

display = PrecisionRecallDisplay.from_estimator(lr, testDf, trueTestDf)
plt.show()

print(precision_recall_curve(trueTestDf, pred_proba_test[:, 1]))

# можно пользоваться RocCurveDisplay
from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(lr, testDf, trueTestDf);
plt.show()

# ROC
fpr, tpr, thresholds = roc_curve(trueTestDf, pred_proba_test[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange',
         lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()

# подбор оптимальной отсечки
precision, recall, thresholds = precision_recall_curve(trueTestDf, pred_proba_test[:, 1])
print(len(precision), len(recall), len(thresholds))

# обработка нулей
inds = np.where(precision != 0)

precision = precision[inds[0]]
recall = recall[inds[0]]
thresholds = thresholds[inds[0] - 1]

fscore = (2 * precision * recall) / (precision + recall)

ix = np.argmax(fscore)
print(f'Best Threshold={thresholds[ix]:.2f}, F-Score={fscore[ix]:.3f}',
      f'Precision={precision[ix]:.3f}, Recall={recall[ix]:.3f}')

pred_test = np.where(pred_proba_test[:, 1] >= 0.34, 1, 0)

print(classification_report(trueTestDf, pred_test))