import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import warnings

from stepic.E_improove_data.home1.DataPipeline import DataPipeline

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 1. Базовое решение
sourceWeatherDf = pd.read_csv("weatherAUS.csv")
print(sourceWeatherDf.head(2))
# print(sourceWeatherDf.shape)

# 1.1. Удалите все пропущенные значения. Вывод: удалилось приблизительно 2/3 части
weatherDf = sourceWeatherDf.dropna()
# print(weatherDf.shape)

# 1.2. Удалите все категориальные. Вывод: удалилось приблизительно 1/3 колонок
weatherDf = weatherDf.dropna()

catColumns = []
numColumns = []

for column in weatherDf.columns:
    if (weatherDf[column].dtypes == 'object' and weatherDf[column].value_counts().shape[0] != 2):
        catColumns.append(column)
    else:
        numColumns.append(column)

weatherDf.drop(columns=catColumns, inplace=True)
# print(weatherDf.shape)

# 1.4. Обучите модель. Вывод: удалилось приблизительно 1/3 колонок
x = weatherDf.drop(columns='RainTomorrow')
y = weatherDf['RainTomorrow']

# разделение
trainDf, testDf, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True, stratify=y)
# print(trainDf)
# преобразование категориальных признаков
catColumns = []
for column in trainDf.columns:
    if (trainDf[column].dtypes == 'object'):
        catColumns.append(column)
for cat_colname in catColumns:
    trainDf = pd.concat([trainDf, pd.get_dummies(
        trainDf[cat_colname],
        prefix=cat_colname,
        dtype='int'
    )], axis=1)
trainDf.drop(columns=catColumns, inplace=True)
for cat_colname in catColumns:
    testDf = pd.concat([testDf, pd.get_dummies(
        testDf[cat_colname],
        prefix=cat_colname,
        dtype='int'
    )], axis=1)
testDf.drop(columns=catColumns, inplace=True)

# масштабирование
scaler = StandardScaler()
# print(trainDf)
trainDf = scaler.fit_transform(trainDf)
testDf = scaler.transform(testDf)

# обучение
model = LogisticRegression()
model.fit(trainDf, yTrain)

predTrain = model.predict(trainDf)
predTest = model.predict(testDf)


# 1.4. Выберете и посчитайте метрику качества
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
    f1_0 = report['No']['f1-score']
    f1_1 = report['Yes']['f1-score']
    return f1_macro, f1_0, f1_1


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


# метрики качества
# f1_macro_train = 0.76299778000628
# f1_macro_test = 0.7634474610899262
# f1_0_train = 0.9094071406072393
# f1_0_test = 0.9094794520547945
# f1_1_train = 0.6165884194053208
# f1_1_test = 0.6174154701250579
print(run_experiment(trainDf, testDf, yTrain, yTest))

# 2. Проведите первичный анализ данных ---------------------------------------------------------------------------------
print(sourceWeatherDf.dtypes)

# проверка на аномалии
print(sourceWeatherDf.describe())

# 3. Проведите визуальный анализ данных

# аномалии в Rainfall с разбивкой на локацию
# вывод: обнаружены аномалии свыше 75
sourceWeatherDf.groupby('Location')['Rainfall'].hist(bins=20)
plt.show()
# обнуляем аномальные данные Rainfall
sourceWeatherDf.loc[sourceWeatherDf['Rainfall'] > 75.0, 'Rainfall'] = np.nan

# аномалии в Evaporation с разбивкой на локацию
# вывод: обнаружены аномалии свыше 85
sourceWeatherDf.groupby('Location')['Evaporation'].hist(bins=20)
plt.show()
# обнуляем аномальные данные Evaporation
sourceWeatherDf.loc[sourceWeatherDf['Evaporation'] > 85.0, 'Evaporation'] = np.nan

# аномалии в Sunshine с разбивкой на локацию
# вывод: аномалий не обнаружено
sourceWeatherDf.groupby('Location')['Sunshine'].hist(bins=20)
plt.show()

# аномалии в WindGustSpeed с разбивкой на локацию
# вывод: обнаружены аномалии свыше 120
sourceWeatherDf.groupby('Location')['WindGustSpeed'].hist(bins=20)
plt.show()
# обнуляем аномальные данные WindGustSpeed
sourceWeatherDf.loc[sourceWeatherDf['WindGustSpeed'] >= 120, 'WindGustSpeed'] = np.nan

# аномалии в WindSpeed9am с разбивкой на локацию
# вывод: обнаружены аномалии свыше 60
sourceWeatherDf.groupby('Location')['WindSpeed9am'].hist(bins=20)
plt.show()
# обнуляем аномальные данные WindGustSpeed
sourceWeatherDf.loc[sourceWeatherDf['WindSpeed9am'] >= 60, 'WindSpeed9am'] = np.nan

# аномалии в WindSpeed3pm с разбивкой на локацию
# вывод: обнаружены аномалии свыше 68
sourceWeatherDf.groupby('Location')['WindSpeed3pm'].hist(bins=20)
plt.show()
# обнуляем аномальные данные WindSpeed3pm
sourceWeatherDf.loc[sourceWeatherDf['WindSpeed3pm'] >= 60, 'WindSpeed3pm'] = np.nan

# аномалии в Humidity9am с разбивкой на локацию
# вывод: аномалий не обнаружено
sourceWeatherDf.groupby('Location')['Humidity9am'].hist(bins=20)
plt.show()

# аномалии в Humidity3pm с разбивкой на локацию
# вывод: аномалий не обнаружено
sourceWeatherDf.groupby('Location')['Humidity3pm'].hist(bins=20)
plt.show()

# аномалии в Pressure9am с разбивкой на локацию
# вывод: аномалий не обнаружено
sourceWeatherDf.groupby('Location')['Pressure9am'].hist(bins=20)
plt.show()

# аномалии в Pressure3pm с разбивкой на локацию
# вывод: аномалий не обнаружено
sourceWeatherDf.groupby('Location')['Pressure3pm'].hist(bins=20)
plt.show()

# аномалии в Cloud9am с разбивкой на локацию
# вывод: аномалий свыше 8.3
sourceWeatherDf.groupby('Location')['Cloud9am'].hist(bins=20)
plt.show()
# обнуляем аномальные данные Cloud9am
sourceWeatherDf.loc[sourceWeatherDf['Cloud9am'] >= 8.3, 'Cloud9am'] = np.nan

# аномалии в Cloud3pm с разбивкой на локацию
# вывод: аномалий свыше 8.3
sourceWeatherDf.groupby('Location')['Cloud3pm'].hist(bins=20)
plt.show()
# обнуляем аномальные данные Cloud3pm
sourceWeatherDf.loc[sourceWeatherDf['Cloud3pm'] >= 8.3, 'Cloud3pm'] = np.nan

# аномалии в Temp9am с разбивкой на локацию
# вывод: аномалий не обнаружено
sourceWeatherDf.groupby('Location')['Temp9am'].hist(bins=20)
plt.show()

# аномалии в Temp3pm с разбивкой на локацию
# вывод: аномалий не обнаружено
sourceWeatherDf.groupby('Location')['Temp3pm'].hist(bins=20)
plt.show()

# матрица корреляций
# вывод: наблюдается большое количество зависимостей
plt.figure(figsize=(20, 20))
corr_matrix = sourceWeatherDf.select_dtypes('number').corr()
corr_matrix = np.round(corr_matrix, 2)
corr_matrix[np.abs(corr_matrix) < 0.1] = 0
sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap='coolwarm')
plt.title('Correlation matrix')
plt.show()

# 4. Разбейте данные на обучение и тест

# удалены целевые значения null
weatherDf = sourceWeatherDf[~(sourceWeatherDf['RainTomorrow'].isna())]
x = weatherDf.drop(columns=['RainTomorrow', 'Date'])
y = weatherDf['RainTomorrow']
trainDf, testDf, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0, shuffle=True, stratify=y)

# 5. Сделайте предобработку данных с помощью класса и пайплайна

pipe = DataPipeline()
trainDf = pipe.fit_transform(trainDf)

numPipe = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

catPipe = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown="ignore")
)

binPipe = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numPipe, pipe.numColumns),
        ("cat", catPipe, pipe.catColumns),
        ("bin", binPipe, pipe.binColumns),
    ]
)

trainDf = preprocessor.fit_transform(trainDf)
testDf = preprocessor.transform(testDf)

# проверка что размерность одинаковая
# print(trainDf.shape, yTrain.shape)

# 6. Обучите модель классификации с целевым признаком RainTomorrow

# NOT ничего не делаем с данными
# Вывод: при удалении null значений и без учета категориальных признаков предсказания модели были лучше
stata, model = run_experiment(trainDf, testDf, yTrain, yTest, method='not')
print(stata)

# OVER меньший класс увеличиваем до большего
TARGET_NAME = 'RainTomorrow'
dfForBalancing = pd.DataFrame(np.c_[trainDf.toarray(), yTrain.map({'Yes': 1, 'No': 0})])  # преобразование в int
TARGET_NUM = 115
dfBalanced = balance_df_by_target(dfForBalancing, TARGET_NUM, method='over')
# print(dfBalanced[TARGET_NUM].value_counts())
trainBalancedDf = dfBalanced.drop(columns=[TARGET_NUM])
yTrainBalanced = dfBalanced[TARGET_NUM]
stata, model = run_experiment(trainBalancedDf, testDf, yTrainBalanced.map({1: 'Yes', 0: 'No'}), yTest, method='over')
print(stata)

# UNDER
dfBalanced = balance_df_by_target(dfForBalancing, TARGET_NUM, method='under')
trainBalancedDf = dfBalanced.drop(columns=[TARGET_NUM])
yTrainBalanced = dfBalanced[TARGET_NUM]
stata, model = run_experiment(trainBalancedDf, testDf, yTrainBalanced.map({1: 'Yes', 0: 'No'}), yTest, method='under')
print(stata)

# TOMEK
dfBalanced = balance_df_by_target(dfForBalancing, TARGET_NUM, method='tomek')
trainBalancedDf = dfBalanced.drop(columns=[TARGET_NUM])
yTrainBalanced = dfBalanced[TARGET_NUM]

stata, model = run_experiment(trainBalancedDf, testDf, yTrainBalanced.map({1: 'Yes', 0: 'No'}), yTest, method='tomek')
print(stata)

# SMOTE - генерация синтетики для заполнения пустот
dfBalanced = balance_df_by_target(dfForBalancing, TARGET_NUM, method='smote')
trainBalancedDf = dfBalanced.drop(columns=[TARGET_NUM])
yTrainBalanced = dfBalanced[TARGET_NUM]
stata, model = run_experiment(trainBalancedDf, testDf, yTrainBalanced.map({1: 'Yes', 0: 'No'}), yTest, method='smote')
print(stata)

# 7. Оцените подходящие метрики качества:
# предсказания с удалением категориальных признаков и Null полей имеет лучший Fscore для тестовых данных = 0.7634474610899262

# 8. Сформулируйте выводы по проделанной работе:
# предсказание что будет дождь Fscore=0.6174154701250579
# предсказание что дождя не будет Fscore=0.9094794520547945
# модель предсказывает что дождя не будет качественнее, чем то что будет дождь
