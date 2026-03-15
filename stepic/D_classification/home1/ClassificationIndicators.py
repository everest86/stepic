import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (roc_auc_score, roc_curve, auc, confusion_matrix, \
                             accuracy_score, classification_report, \
                             precision_recall_curve, recall_score, precision_score, fbeta_score, f1_score)

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
print('Precision fun = ', precision_score(y, b))
print('Precision fun for 1 = ', precision_score(y, b, average='binary', pos_label=1))

recall = TP / (TP + FN)
print("Recall = ", recall)
print('Recall fun = ', recall_score(y, b))
print('Recall fun for 1 = ', recall_score(y, b, average='binary', pos_label=1))

F = 2 * (precision * recall) / (precision + recall)
print('F-score = ', F)
print('F-score fun = ', f1_score(y_true=y, y_pred=b))
print('F-score beta=1 fun = ', fbeta_score(y_true=y, y_pred=b, beta=1))

print(classification_report(y, b))

# 4. ROC
y = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 0])
b = np.array([0.1, 0.8, 0.45, 0.23, 0.66, 0.12, 0.69, 0.33, 0.92, 0.13])

# UPD 2025-07-29
# plot_roc_curve больше нет в sklearn
# plot_roc_curve(lr, X_test, y_test);

# можно пользоваться RocCurveDisplay

# fpr, tpr, thresholds = roc_curve(y, b[:, 1])
# roc_auc = auc(fpr, tpr)
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange',
#          lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC curve')
# plt.legend(loc="lower right")
# plt.show()
