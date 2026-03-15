import pandas as pd
from sklearn.metrics import (roc_auc_score, roc_curve, auc, confusion_matrix, \
                             accuracy_score, classification_report, \
                             precision_recall_curve, recall_score, precision_score, fbeta_score)

y = [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0]
b = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]

TP = 5  # единицы совпадают
FP = 1  # модель предсказала кота, на деле голубь
FN = 3  # модель предсказала голубя, на деле кот
TN = 12  # модель правильно предсказала голубя

# 1. Подсчет accuracy = 0.8095238095238095
accuracy = (TP + TN) / (TP + TN + FP + FN)
print(accuracy, accuracy_score(y, b))  # проверка что accuracy_score(y, b) тоже равен 0.8095238095238095

# 2. Матрица ошибок
print(pd.crosstab(y, b))

# 3. precision, recall, F-score
precision = TP / (TP + FP)
print('Precision = ', precision)
print('Precision fun = ', precision_score(b, y))

recall = TP / (TP + FN)
print("Recall = ", recall)
print('Recall fun = ', recall_score(b, y))

F = 2 * (precision * recall) / (precision + recall)
print('F-score = ', F)
print('F-score fun = ', fbeta_score(y_true=y, y_pred=b, beta=1))

# 4. ROC
TPR = TP / (TP + FP)
print('TPR = ', TPR)
FPR = FP / (FP + TN)
print('FPR = ', FPR)
