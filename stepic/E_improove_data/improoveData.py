import pandas as pd
import numpy as np
import warnings
import imblearn

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

empDf = pd.read_csv("employee.csv")
print(empDf.head(2))

# круговая диаграмма покинувших контору
empDf['left'].value_counts().plot(kind='pie', autopct='%.2f%%')
plt.show()

# разделение данных
x = empDf.drop(columns='left')
y = empDf['left']

trainDf, testDf, yTrain, yTest = train_test_split(x, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)

print(trainDf.shape, testDf.shape)

# проверка на null
# print(trainDf.isnull().sum())

# разбивка на типы
num_features = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']
bin_features = ['Work_accident', 'promotion_last_5years']
cat_features = ['department', 'salary']

# замена null значений
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_pipe = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

cat_pipe = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown="ignore")
)

bin_pipe = make_pipeline(
    SimpleImputer(strategy='most_frequent')
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features),
        ("bin", bin_pipe, bin_features)
    ]
)

trainDf = preprocessor.fit_transform(trainDf)
testDf = preprocessor.transform(testDf)


# Подготовка
# функция для сэмплирования при несбалансированных данных
def balance_df_by_target(df, target_idx, method='over'):
    assert method in ['over', 'under', 'tomek', 'smote'], 'Неверный метод сэмплирования'

    df[target_idx] = df[target_idx].astype('int')
    target_counts = df[target_idx].value_counts()

    major_class_name = target_counts.argmax()
    minor_class_name = target_counts.argmin()

    disbalance_coeff = int(target_counts[major_class_name] / target_counts[minor_class_name]) - 1
    if method == 'over':
        for i in range(disbalance_coeff):
            sample = df[df[target_idx] == minor_class_name].sample(target_counts[minor_class_name])
            # UPD 2025-07-30
            # метода .append больше нет в pandas
            # df = df.append(sample, ignore_index=True)

            # можно заменить на .concat
            df = pd.concat([df, sample], ignore_index=True)

    elif method == 'under':
        df_ = df.copy()
        df = df_[df_[target_idx] == minor_class_name]
        tmp = df_[df_[target_idx] == major_class_name]
        # UPD 2025-07-30
        # метода .append больше нет в pandas
        # df = df.append(tmp.iloc[
        #     np.random.randint(0, tmp.shape[0], target_counts[minor_class_name])
        # ], ignore_index=True)

        # можно заменить на .concat
        df = pd.concat(
            [df, tmp.iloc[
                np.random.randint(0, tmp.shape[0], target_counts[minor_class_name])
            ]
             ], ignore_index=True)

    elif method == 'tomek':
        from imblearn.under_sampling import TomekLinks
        tl = TomekLinks()
        X_tomek, y_tomek = tl.fit_resample(df.drop(columns=target_idx), df[target_idx])
        df = pd.concat([X_tomek, y_tomek], axis=1)

    elif method == 'smote':
        from imblearn.over_sampling import SMOTE
        smote = SMOTE()
        X_smote, y_smote = smote.fit_resample(df.drop(columns=target_idx), df[target_idx])
        df = pd.concat([X_smote, y_smote], axis=1)

    return df.sample(frac=1)


def get_metrics(report):
    f1_macro = report['macro avg']['f1-score']
    f1_0 = report['0']['f1-score']
    f1_1 = report['1']['f1-score']
    return f1_macro, f1_0, f1_1


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def run_experiment(X_train, X_test, y_train, y_test, method='not'):
    assert method in ['not', 'over', 'under', 'tomek', 'smote'], 'Неправильный метод сэмплирования'

    model = LogisticRegression()
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    report_train = classification_report(y_train, pred_train, output_dict=True)
    report_test = classification_report(y_test, pred_test, output_dict=True)
    f1_macro_train, f1_0_train, f1_1_train = get_metrics(report_train)
    f1_macro_test, f1_0_test, f1_1_test = get_metrics(report_test)

    stata = {
        f'f1_macro_train': f1_macro_train,
        f'f1_macro_test': f1_macro_test,
        f'f1_0_train': f1_0_train,
        f'f1_0_test': f1_0_test,
        f'f1_1_train': f1_1_train,
        f'f1_1_test': f1_1_test,
        f'method': method
    }

    return stata, model


# ----------------------------------------- Генерация-добавление строк------------------
# NOT ничего не делаем с данными
stata, model = run_experiment(trainDf, testDf, yTrain, yTest, method='not')
print(stata)

# OVER меньший класс увеличиваем до большего
TARGET_NAME = 'left'

dfForBalancing = pd.DataFrame(np.c_[trainDf, yTrain.values])
# print(dfForBalancing.head(2))

TARGET_NAME = 20
dfBalanced = balance_df_by_target(dfForBalancing, TARGET_NAME, method='over')
# print(dfBalanced)
print(dfBalanced[TARGET_NAME].value_counts())

trainBalancedDf = dfBalanced.drop(columns=[TARGET_NAME])
yTrainBalanced = dfBalanced[TARGET_NAME]

stata, model = run_experiment(trainBalancedDf, testDf, yTrainBalanced, yTest, method='over')
print(stata)

# UNDER
dfBalanced = balance_df_by_target(dfForBalancing, TARGET_NAME, method='under')
print(dfBalanced[TARGET_NAME].value_counts())
trainBalancedDf = dfBalanced.drop(columns=[TARGET_NAME])
yTrainBalanced = dfBalanced[TARGET_NAME]

stata, model = run_experiment(trainBalancedDf, testDf, yTrainBalanced, yTest, method='under')
print(stata)

# TOMEK
dfBalanced = balance_df_by_target(dfForBalancing, TARGET_NAME, method='tomek')
print(dfBalanced[TARGET_NAME].value_counts())
trainBalancedDf = dfBalanced.drop(columns=[TARGET_NAME])
yTrainBalanced = dfBalanced[TARGET_NAME]

stata, model = run_experiment(trainBalancedDf, testDf, yTrainBalanced, yTest, method='tomek')
print(stata)

# SMOTE - генерация синтетики для заполнения пустот
dfBalanced = balance_df_by_target(dfForBalancing, TARGET_NAME, method='smote')
print(dfBalanced[TARGET_NAME].value_counts())
trainBalancedDf = dfBalanced.drop(columns=[TARGET_NAME])
yTrainBalanced = dfBalanced[TARGET_NAME]

stata, model = run_experiment(trainBalancedDf, testDf, yTrainBalanced, yTest, method='smote')
print(stata)

# ------------------ Работа с null ячейками -------------------

print(empDf.isnull().sum())
print(empDf.shape)

# удалить строки - потеря данных
print(empDf.dropna().isnull().sum())
print(empDf.dropna().shape)

# удалить столбцов - если больше половины пропусков от общего - можно удалить столбец
empDf.drop(columns='last_evaluation')

# замена на среднее, медиану или моду
median = empDf['last_evaluation'].median()
# заполение пустых значений на медиану
empDf['last_evaluation'].fillna(median)

# замена бинарного признака на моду
mode = empDf['Work_accident'].mode()[0]
print(empDf['Work_accident'].fillna(mode).isnull().sum())

# замена на медиану (с группировкой)
# пропуски могут остаться, но их после можно заменить на медиану
empDf[['number_project', 'average_montly_hours']].head()
empDf.groupby('number_project')['average_montly_hours'].median()
transform_med = empDf.groupby('number_project')['average_montly_hours'].transform('median')
print(transform_med)
print(empDf['average_montly_hours'].fillna(transform_med).isnull().sum())

# пометка пропусков
print(empDf['salary'].isna().sum())
# новый признак
empDf['salary_nan'] = 0
empDf.loc[empDf['salary'].isna(), 'salary_nan'] = 1

# замена категориальных на что-то другое
empDf['salary'].fillna('unknown', inplace=True)

# KNNImputer - заполнение пропусков на модели машиного обучения
#  смотрит на ближайших соседей
from sklearn.impute import KNNImputer

knn = KNNImputer(n_neighbors=5)
knn.fit(empDf[num_features])

# --------------------- ВЫБРОСЫ ------------------
houseDf = pd.read_csv("california_housing_train.csv")
print(houseDf.head(2))

# визуально (гистограммы, боксплот)
houseDf.hist(figsize=(16, 16))
plt.show()
# аномалии
print(houseDf[houseDf['total_rooms'] > 20000].shape)
houseDf = houseDf[houseDf['total_rooms'] < 20000]
houseDf['total_rooms'].hist()
plt.show()
houseDf['households'].plot(kind='box')
plt.show()

houseDf[houseDf['households'] > 1500].shape
df = houseDf[houseDf['households'] < 1500]
df['households'].plot(kind='box')
plt.show()

# можно выбросы пометить как NaN, далее произвести заполнение null данных
houseDf.loc[houseDf['total_rooms'] > 20000, 'total_rooms'] = np.nan

# подсчетом

# межквартильный размах
print(houseDf['total_bedrooms'].describe())
q1 = houseDf['total_bedrooms'].quantile(0.25)
q3 = houseDf['total_bedrooms'].quantile(0.75)
iqr = 1.5 * (q3 - q1)
left = q1 - iqr
right = q3 + iqr
print(houseDf[~(houseDf['total_bedrooms'].between(left, right))])

# правило трех сигм
s = houseDf['total_bedrooms'].std()  # стандартное отклонение
left = -3 * s + houseDf['total_bedrooms'].mean()
right = 3 * s + houseDf['total_bedrooms'].mean()
print(left, right)
print(houseDf[~(houseDf['total_bedrooms'].between(left, right))])

# Методы машинного обучения
# OneClassSVM - линейная модель максимизирует расстояние между классами. чем ближе к началу координат - выброс, чем дальше то норм
df.plot(kind='scatter', x='total_rooms', y='total_bedrooms')
plt.show()
from sklearn.svm import OneClassSVM

X = StandardScaler().fit_transform(df[['total_rooms', 'total_bedrooms']])
clf = OneClassSVM(nu=0.01)  # ищет 1%
clf.fit(X)
yPred = clf.predict(houseDf[['total_rooms', 'total_bedrooms']])  # 1 - объект норм, -1 - выброс
print(houseDf[yPred == -1])

import matplotlib.pyplot as plt

xx, yy = np.meshgrid(np.linspace(-1000, houseDf['total_rooms'].max(), 500),
                     np.linspace(-1000, houseDf['total_bedrooms'].max(), 500))

Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 9))
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

s = 40
b1 = plt.scatter(houseDf['total_rooms'], houseDf['total_bedrooms'], c='white', s=s, edgecolors='k')

plt.legend([a.legend_elements()[0][0], b1],
           ["разделяющая граница", "обучающие данные"],
           loc="upper left");
plt.show()

# Isolation forest
X = StandardScaler().fit_transform(df[['total_rooms', 'total_bedrooms']])
from sklearn.ensemble import IsolationForest

clf = IsolationForest(random_state=1, contamination=0.01)  # 1% выбросов
clf.fit(X)

y_pred = clf.predict(houseDf[['total_rooms', 'total_bedrooms']])  # 1 - объект норм, -1 - выброс
print(y_pred)

import matplotlib.pyplot as plt

xx, yy = np.meshgrid(np.linspace(-1000, houseDf['total_rooms'].max(), 500),
                     np.linspace(-1000, houseDf['total_bedrooms'].max(), 500))

Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(12, 9))
plt.title("Выявление выбросов на искусственных данных")

a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

s = 40
b1 = plt.scatter(houseDf['total_rooms'], houseDf['total_bedrooms'], c='white', s=s, edgecolors='k')

plt.legend([a.legend_elements()[0][0], b1],
           ["разделяющая граница", "обучающие данные"],
           loc="upper left");
plt.show()

# KNN - смотрит на расстояние
X = StandardScaler().fit_transform(df[['total_rooms', 'total_bedrooms']])
from sklearn.neighbors import LocalOutlierFactor

clf = LocalOutlierFactor(novelty=True)
clf.fit(X)

y_pred = clf.predict(houseDf[['total_rooms', 'total_bedrooms']])
print(y_pred)
import matplotlib.pyplot as plt

xx, yy = np.meshgrid(np.linspace(-1000, houseDf['total_rooms'].max(), 500),
                     np.linspace(-1000, houseDf['total_bedrooms'].max(), 500))

Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(12, 9))
plt.title("Выявление выбросов на искусственных данных")

a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

s = 40
b1 = plt.scatter(houseDf['total_rooms'], houseDf['total_bedrooms'], c='white', s=s, edgecolors='k')

plt.legend([a.legend_elements()[0][0], b1],
           ["разделяющая граница", "обучающие данные"],
           loc="upper left");
plt.show()

# DBSCAN
X = StandardScaler().fit_transform(df[['total_rooms', 'total_bedrooms']])
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.5, min_samples=5).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
# Количество кластеров в метках без учета шума
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Кол-во кластеров', n_clusters_)
print('Кол-во шума', n_noise_)

# Отрисовка результата
plt.figure(figsize=(12, 9))
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Черный цвет используется для выявления шума
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Оценочное кол-во кластеров: %d' % n_clusters_)
plt.show()
