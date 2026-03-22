import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
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

# 1.1. Сделайте минимальные преобразования
# Масштабирование
scaler = StandardScaler()
trainDf = scaler.fit_transform(trainDf)
testDf = scaler.transform(testDf)

# 1.2. Обучение
model = LogisticRegression()
model.fit(trainDf, yTrain)
print(model.coef_)

predTrain = model.predict(trainDf)
predTest = model.predict(testDf)

# print(classification_report(yTrain, predTrain))
print(classification_report(yTest, predTest))

# 1.3. Выберете и посчитайте метрику качества
# Вывод: для тестовых данных предсказание что это мошенник f1-score=0.73

# 2. Разбейте данные на обучение и тест
x = sourceDf.drop(columns=['Class'])
y = sourceDf['Class']

trainDf, testDf, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y, shuffle=True)


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


# Масштабирование
scaler = StandardScaler()
trainDf = scaler.fit_transform(trainDf)
testDf = scaler.transform(testDf)

# NOT ничего не делаем с данными
# Вывод: при удалении null значений и без учета категориальных признаков предсказания модели были лучше
stata, model = run_experiment(trainDf, testDf, yTrain, yTest, method='not')
print(stata)

# 3. Проведите балансировку данных минимум тремя методами

# OVER меньший класс увеличиваем до большего
TARGET_NAME = 'Class'
dfForBalancing = pd.DataFrame(np.c_[trainDf, yTrain])  # преобразование в int
# print(dfForBalancing)
TARGET_NUM = 30
dfBalanced = balance_df_by_target(dfForBalancing, TARGET_NUM, method='over')
# print(dfBalanced[TARGET_NUM].value_counts())
trainBalancedDf = dfBalanced.drop(columns=[TARGET_NUM])
yTrainBalanced = dfBalanced[TARGET_NUM]
stata, model = run_experiment(trainBalancedDf, testDf, yTrainBalanced, yTest, method='over')
print(stata)

# UNDER
dfBalanced = balance_df_by_target(dfForBalancing, TARGET_NUM, method='under')
trainBalancedDf = dfBalanced.drop(columns=[TARGET_NUM])
yTrainBalanced = dfBalanced[TARGET_NUM]
stata, model = run_experiment(trainBalancedDf, testDf, yTrainBalanced, yTest, method='under')
print(stata)

# TOMEK
dfBalanced = balance_df_by_target(dfForBalancing, TARGET_NUM, method='tomek')
trainBalancedDf = dfBalanced.drop(columns=[TARGET_NUM])
yTrainBalanced = dfBalanced[TARGET_NUM]

stata, model = run_experiment(trainBalancedDf, testDf, yTrainBalanced, yTest, method='tomek')
print(stata)

# SMOTE - генерация синтетики для заполнения пустот
dfBalanced = balance_df_by_target(dfForBalancing, TARGET_NUM, method='smote')
trainBalancedDf = dfBalanced.drop(columns=[TARGET_NUM])
yTrainBalanced = dfBalanced[TARGET_NUM]
stata, model = run_experiment(trainBalancedDf, testDf, yTrainBalanced, yTest, method='smote')
print(stata)

# 6. Сформулируйте выводы по проделанной работе
# для предсказания мошеннических транзакция сэмплирование tomek показало наилучший f1 = 0.7413127413127413
