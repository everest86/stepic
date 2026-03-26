import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from onnx.reference.ops.aionnxml.op_one_hot_encoder import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (LogisticRegression, LinearRegression)
from sklearn.model_selection import train_test_split
import warnings
from sklearn.preprocessing import LabelEncoder
from category_encoders.count import CountEncoder
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
print(sourceDf.isnull().sum())  # пропусков нет

print(sourceDf.dtypes)  # типы данных

print(sourceDf.describe())

print(sourceDf[targetColumn].value_counts())  # целевые значения имеет 2 уникальных значения

num_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']  # вещественные поля

df = sourceDf[num_columns + [targetColumn]]

x = df[num_columns]
y = df[targetColumn]

# разделение данных
trainDf, testDf, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)

# масштабирование
scaler = StandardScaler()
trainDf = scaler.fit_transform(trainDf)
testDf = scaler.transform(testDf)

# обучение
model = LogisticRegression()
model.fit(trainDf, yTrain)

# получение предсказаний
testPred = model.predict(testDf)

# 0 precision = 0.83    recall =  0.94   f1 =   0.89
# 1 precision = 0.70    recall =  0.40   f1 =   0.51
# для ЗП <= 50 - precision (в выборку попадут только с ЗП <= 50) = 0.83%, recall (сколько % от общего кол-ва с ЗП <= 50) == 0.94 (т.е. 6% скорее не угадаем)
# для ЗП > 50 - precision (в выборку попадут только с ЗП <= 50) = 0.70%, recall (сколько % от общего кол-ва с ЗП <= 50) == 0.40 (т.е. 60% скорее не угадаем)
# Вывод: на ограниченном наборе данных точность предсказаний мала
print(classification_report(yTest, testPred))

# Accuracy - достаточно высокий, но он не показателен. Т.к. имеется большая разностьв количестве с зп == <=50K в 3 раза больше
print(model.score(testDf, yTest))

# 2. Проведите первичный и визуальный анализ данных
empDf = sourceDf.copy()

targetColumn = 'salary'
numColumns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
catColumns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

# проверка workclass на уникальные значения
# вывод: найдено 9 уникальных
print(empDf['workclass'].value_counts().count())

# проверка education на уникальные значения
# вывод: найдено 16 уникальных
print(empDf['education'].value_counts().count())

# проверка marital-status на уникальные значения
# вывод: найдено 7 уникальных
print(empDf['marital-status'].value_counts().count())

# проверка occupation на уникальные значения
# вывод: найдено 15 уникальных
print(empDf['occupation'].value_counts().count())

# проверка relationship на уникальные значения
# вывод: найдено 6 уникальных
print(empDf['relationship'].value_counts().count())

# проверка race на уникальные значения
# вывод: найдено 5 уникальных
print(empDf['race'].value_counts().count())

# проверка sex на уникальные значения
# вывод: найдено 2 уникальных
print(empDf['sex'].value_counts().count())

# вызуальный анализ
# print(sourceDf.describe())

# график зависимости salary от education
# вывод: у докторов и профи зарплата выше остальных
empDf[['education', targetColumn]].groupby('education')['salary'].mean().plot(kind='bar')
plt.xlabel('Education')
plt.ylabel('Average Salary')
plt.title('Average Salary by Education')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# график зависимости salary от marital-status
# вывод: у женатых военнослужащих зарплата выше остальных
empDf[['marital-status', targetColumn]].groupby('marital-status')['salary'].mean().plot(kind='bar')
plt.xlabel('marital-status')
plt.ylabel('Average Salary')
plt.title('Average Salary by marital-status')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# график зависимости salary от workclass
# вывод: у предпринимателя зарплата выше остальных
empDf[['workclass', targetColumn]].groupby('workclass')['salary'].mean().plot(kind='bar')
plt.xlabel('workclass')
plt.ylabel('Average Salary')
plt.title('Average Salary by workclass')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 3. Сделайте предобработку данных
# целевой признак заменен на 0 и 1

# 4. Обучите модель классификации с целевым признаком salary
# x = sourceDf.drop(columns=[targetColumn])
# y = sourceDf[targetColumn]
# trainDf, testDf, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y, shuffle=True)
#
# # масштабирование вещественных признаков
# scaler = StandardScaler()
# trainDf[numColumns] = scaler.fit_transform(trainDf[numColumns])
# testDf[numColumns] = scaler.transform(testDf[numColumns])

def getFscore(df):
    x = df.drop(columns=[targetColumn])
    y = df[targetColumn]
    trainDf, testDf, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y, shuffle=True)

    scaler = StandardScaler()
    trainDf = scaler.fit_transform(trainDf)
    testDf = scaler.transform(testDf)

    model = LogisticRegression()
    model.fit(trainDf, yTrain)

    predTest = model.predict(testDf)

    report = classification_report(yTest, predTest, output_dict=True)

    print(report['macro avg']['f1-score'])

    return model


# кодирование категориальных признаков
def get_one_hot(X, cols):
    for each in cols:
        dummies = pd.get_dummies(X[each], prefix=each, dtype='int')
        X = pd.concat([X, dummies], axis=1)
    return X


def get_label(X, cols):
    X = X.copy()
    for each in cols:
        le = LabelEncoder()
        labels = le.fit_transform(X[each])
        X[each] = labels
    return X


def get_count(X, cols):
    X = X.copy()

    for col in cols:
        X[col] = X[col].astype('str')

    ce = CountEncoder(handle_unknown=-1)
    ce.fit(X[cols])
    X[cols] = ce.transform(X[cols])
    return X


print("Encoding column: sex")
# пробуем закодировать oneHotEncoding
tmpDf = get_one_hot(empDf, ['sex']).drop(columns=catColumns)
getFscore(tmpDf)
# пробуем закодировать countEncoding
tmpDf = get_count(empDf, ['sex']).drop(columns=catColumns)
getFscore(tmpDf)
# лучше метрика на oneHotEncoding
empDf = get_one_hot(empDf, ['sex']).drop(columns='sex')
catColumns.remove('sex')

print("Encoding column: native-country")
# пробуем закодировать oneHotEncoding
tmpDf = get_one_hot(empDf, ['native-country']).drop(columns=catColumns)
getFscore(tmpDf)
# пробуем закодировать countEncoding
tmpDf = get_count(empDf, ['native-country']).drop(columns=catColumns)
getFscore(tmpDf)
# лучше метрика на oneHotEncoding
empDf = get_one_hot(empDf, ['native-country']).drop(columns='native-country')
catColumns.remove('native-country')

print("Encoding column: race")
# пробуем закодировать oneHotEncoding
tmpDf = get_one_hot(empDf, ['race']).drop(columns=catColumns)
getFscore(tmpDf)
# пробуем закодировать countEncoding
tmpDf = get_count(empDf, ['race']).drop(columns=catColumns)
getFscore(tmpDf)
# лучше метрика на oneHotEncoding
empDf = get_one_hot(empDf, ['race']).drop(columns='race')
catColumns.remove('race')

print("Encoding column: relationship")
# пробуем закодировать oneHotEncoding
tmpDf = get_one_hot(empDf, ['relationship']).drop(columns=catColumns)
getFscore(tmpDf)
# пробуем закодировать countEncoding
tmpDf = get_count(empDf, ['relationship']).drop(columns=catColumns)
getFscore(tmpDf)
# лучше метрика на oneHotEncoding
empDf = get_one_hot(empDf, ['relationship']).drop(columns='relationship')
catColumns.remove('relationship')

print("Encoding column: occupation")
# пробуем закодировать oneHotEncoding
tmpDf = get_one_hot(empDf, ['occupation']).drop(columns=catColumns)
getFscore(tmpDf)
# пробуем закодировать countEncoding
tmpDf = get_count(empDf, ['occupation']).drop(columns=catColumns)
getFscore(tmpDf)
# лучше метрика на oneHotEncoding
empDf = get_one_hot(empDf, ['occupation']).drop(columns='occupation')
catColumns.remove('occupation')

print("Encoding column: marital-status")
# пробуем закодировать oneHotEncoding
tmpDf = get_one_hot(empDf, ['marital-status']).drop(columns=catColumns)
getFscore(tmpDf)
# пробуем закодировать countEncoding
tmpDf = get_count(empDf, ['marital-status']).drop(columns=catColumns)
getFscore(tmpDf)
# лучше метрика на oneHotEncoding
empDf = get_one_hot(empDf, ['marital-status']).drop(columns='marital-status')
catColumns.remove('marital-status')

print("Encoding column: education")
# пробуем закодировать oneHotEncoding
tmpDf = get_one_hot(empDf, ['education']).drop(columns=catColumns)
getFscore(tmpDf)
# пробуем закодировать countEncoding
tmpDf = get_count(empDf, ['education']).drop(columns=catColumns)
getFscore(tmpDf)
# лучше метрика на count
empDf = get_count(empDf, ['education']).drop(columns='education')
catColumns.remove('education')

print("Encoding column: workclass")
# пробуем закодировать oneHotEncoding
tmpDf = get_one_hot(empDf, ['workclass']).drop(columns=catColumns)
getFscore(tmpDf)
# пробуем закодировать countEncoding
tmpDf = get_count(empDf, ['workclass']).drop(columns=catColumns)
getFscore(tmpDf)
# лучше метрика на oneHotEncoding
empDf = get_one_hot(empDf, ['workclass']).drop(columns='workclass')
catColumns.remove('workclass')


# Вывод: при кодировании категориальных признаков качество предсказаний увеличивается

# 5. Проведите отбор признаков минимум с помощью трех подходов
def get_score(X, y, random_seed=0, model=None, is_return=False):
    if model is None:
        model = LogisticRegression()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y, shuffle=True)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    if is_return:
        print(report['macro avg']['f1-score'])
        return model

    print(report['macro avg']['f1-score'])


x = empDf.drop(columns=[targetColumn])
y = sourceDf[targetColumn]

# одномерный f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# выбрать 25 лучших
# вывод: качество предсказаний уменьшилось из-за уменьшения количества признаков
selector = SelectKBest(f_regression, k=30)
x_new = selector.fit_transform(x, y)
model = get_score(x_new, y, is_return=True)

# одномерный mutual_info_regression
# вывод: качество предсказаний уменьшилось из-за уменьшения количества признаков
from sklearn.feature_selection import mutual_info_regression

selector = SelectKBest(mutual_info_regression, k=30)
x_new = selector.fit_transform(x, y)
model = get_score(x_new, y, is_return=True)

# проценты
from sklearn.feature_selection import SelectPercentile

selector = SelectPercentile(f_regression, percentile=70)
X_new = selector.fit_transform(x, y)
model = get_score(x_new, y, is_return=True)

# Рекурсивный отбор
# вывод: качество предсказаний уменьшилось из-за уменьшения количества признаков
from sklearn.feature_selection import RFE

selector = RFE(model, n_features_to_select=30, step=1)
selector = selector.fit(X_new, y)
x_new = X_new[:, selector.support_]
model = get_score(x_new, y, is_return=True)

# SelectFromModel
# вывод: качество предсказаний уменьшилось из-за уменьшения количества признаков
from sklearn.feature_selection import SelectFromModel

selector = SelectFromModel(model, prefit=False, max_features=30, threshold=-np.inf)
x_new = selector.fit_transform(x, y)
model = get_score(x_new, y, is_return=True)

# Переборный отбор - самый тяжелый, но метрика лучше
from sklearn.feature_selection import SequentialFeatureSelector

sfs_forward = SequentialFeatureSelector(
    model, n_features_to_select=30, direction="forward"
)
sfs_forward.fit(x, y)
x_new = x[:, sfs_forward.get_support()]
model = get_score(x_new, y, is_return=True)

# 6. Оцените подходящие метрики качества
# с уменьшением количества колонок искуственно созданных - уменьшается качество предсказаний

# 7. Сформулируйте выводы по проделанной работе
# с конструированием новых признаков - качество предсказаний увеличивается
# для каждого признака требуется искать наилучший способ кодирования, сделать это можно с помощью метрик
