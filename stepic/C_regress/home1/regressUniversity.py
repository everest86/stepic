import pandas as pd
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from stepic.C_regress.DataPipeline import DataPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse, r2_score as r2

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def evaluate_preds(true_values, pred_values, save=False):
    print("R2:\t" + str(round(r2(true_values, pred_values), 3)) + "\n" +
          "RMSE:\t" + str(round(np.sqrt(mse(true_values, pred_values)), 3)) + "\n" +
          "MSE:\t" + str(round(mse(true_values, pred_values), 3))
          )

    plt.figure(figsize=(8, 8))

    sns.scatterplot(x=pred_values, y=true_values)
    plt.plot([0, 500000], [0, 500000], linestyle='--', color='black')  # диагональ, где true_values = pred_values

    plt.xlabel('Predicted values')
    plt.ylabel('True values')
    plt.title('True vs Predicted values')

    if save == True:
        plt.savefig('report.png')
    plt.show()


# 1. Первичный анализ
admDf = pd.read_csv("adm_data.csv")

# Удалить ненужные поля
admDf.drop(columns='Serial No.', inplace=True)
print(admDf.head(2))

# Оценка количества пустых значений
print(admDf.isna().sum())

print(admDf.describe())

# 2. Визуальный анализ данных. Построение матрицы корреляций.
# Вывод: все параметры влияют на шансы поступления
corr_matrix = admDf.corr()
corr_matrix = np.round(corr_matrix, 2)
corr_matrix[np.abs(corr_matrix) < 0.3] = 0
sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap='coolwarm')
plt.title('Correlation matrix')
plt.show()

# Проверка на аномалии целевого значения
# Вывод: аномалий нет
admDf['Chance of Admit '].hist(bins=20)
plt.show()

# 3. Разбить данные на обучение и тест
x = admDf.drop(columns='Chance of Admit ')
y = admDf['Chance of Admit ']
xOrigTrain, xOrigTest, yTran, yTest = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=2)

scaler = StandardScaler()
pipe = make_pipeline(
    DataPipeline(),
    SimpleImputer(strategy='median'),
    PolynomialFeatures(interaction_only=True),
    scaler
)
x_train = pipe.fit_transform(xOrigTrain)
x_test = pipe.transform(xOrigTest)

model = LinearRegression()
model.fit(x_train, yTran)
y_train_pred = model.predict(x_train)
# Постпроцессинг
y_train_pred = np.clip(y_train_pred, a_min=10000, a_max=500000)
evaluate_preds(yTran, y_train_pred)  # вывод: R стало больше
