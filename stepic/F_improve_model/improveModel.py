import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import (roc_auc_score, roc_curve, auc, confusion_matrix,
                             accuracy_score, classification_report,
                             precision_recall_curve, recall_score, precision_score, fbeta_score, f1_score)

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
data = pd.DataFrame(
    {'date': [
        '2020-05-14',
        '2022-08-23',
        '2019-01-01'
    ]})

# print(data['date'])
# делаем тип дата
data['date'] = pd.to_datetime(data['date'])
# print(data['date'])

# извлекаем новые значения
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['weekday'] = data['date'].dt.weekday

# print(data)

# ## Конструирование признаков
data = pd.read_csv("data.csv")
print(data.head(1))


def get_score(X, y, random_seed=42, model=None, is_return=False):
    if model is None:
        model = LinearRegression()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    if is_return:
        print(model.score(X_test, y_test))
        return model

    return model.score(X_test, y_test)


from sklearn.preprocessing import LabelEncoder
from category_encoders.count import CountEncoder


def get_one_hot(X, cols):
    for each in cols:
        dummies = pd.get_dummies(X[each], prefix=each)
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


# базовое решение
columns = [
    'bathrooms',
    'bedrooms',
    'finishedsqft',
    'latitude',
    'longitude',
    'totalrooms'
]
print(get_score(data[columns], data['zestimate']))

exclude_columns = ['z_address', 'lastsoldprice', 'zestimate', 'lastsolddate', 'neighborhood', 'usecode']

# usecode
print(data['usecode'].value_counts())
tmp = get_one_hot(data, cols=['usecode'])
print(tmp.head())

print(get_score(tmp.drop(columns=exclude_columns), data['zestimate']))

tmp = get_count(data, cols=['usecode'])
print(tmp.head())

exclude_columns_usecode = exclude_columns.copy()
exclude_columns_usecode.remove('usecode')
print(exclude_columns_usecode)

print(get_score(tmp.drop(columns=exclude_columns_usecode), data['zestimate']))

data_processed = get_one_hot(data, cols=['usecode'])

tmp = get_one_hot(data_processed, cols=['neighborhood'])
print(tmp.head())
print(get_score(tmp.drop(columns=exclude_columns), data_processed['zestimate']))

print(data_processed['neighborhood'].value_counts())

tmp = get_count(data_processed, cols=['neighborhood'])
print(tmp.head())

exclude_columns_tmp = exclude_columns.copy()
exclude_columns_tmp.remove('neighborhood')
print(exclude_columns_tmp)

print(get_score(tmp.drop(columns=exclude_columns_tmp), data_processed['zestimate']))

data_processed = get_one_hot(data_processed, cols=['neighborhood'])

print(data_processed['lastsolddate'].head())

data_processed['lastsolddate'] = pd.to_datetime(data_processed['lastsolddate'])

print(data_processed['lastsolddate'].head())

data_processed['lastsoldmonth'] = data_processed['lastsolddate'].dt.month
print(data_processed['lastsoldmonth'])

data_processed['lastsolddate'] = [t.timestamp() for t in data_processed['lastsolddate']]
print(data_processed['lastsolddate'])

exclude_columns_tmp = exclude_columns.copy()
exclude_columns_tmp.remove('lastsolddate')
print(exclude_columns_tmp)

print(get_score(data_processed.drop(columns=exclude_columns_tmp), data_processed['zestimate']))

tmp = get_one_hot(data_processed, cols=['lastsoldmonth']).drop(columns=exclude_columns)
print(tmp.head())
print(get_score(tmp, data_processed['zestimate']))

tmp = get_one_hot(data_processed[['lastsoldmonth']], cols=['lastsoldmonth'])
print(tmp.head())

print(get_score(tmp, data_processed['zestimate']))

print(data_processed['lastsoldmonth'].value_counts())

tmp = get_count(data_processed, cols=['lastsoldmonth']).drop(columns=exclude_columns)
print(tmp.head())
print(get_score(tmp, data_processed['zestimate']))
data_processed = get_count(data_processed, cols=['lastsoldmonth'])

# zipcode обработка
tmp = get_one_hot(data_processed, cols=['zipcode']).drop(columns=exclude_columns)
print(tmp.head())
print(get_score(tmp, data_processed['zestimate']))

tmp = get_label(data_processed, cols=['zipcode']).drop(columns=exclude_columns)
print(tmp.head())
print(get_score(tmp, data_processed['zestimate']))

tmp = get_count(data_processed, cols=['zipcode']).drop(columns=exclude_columns)
tmp.head()
print(get_score(tmp, data_processed['zestimate']))

data_processed = get_one_hot(data_processed, cols=['zipcode'])

# z_address
print(data_processed.head())

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

tfidf = TfidfVectorizer(max_features=50, stop_words='english')
vectorized = tfidf.fit_transform(data_processed['z_address'])

print(vectorized)

vectorized_df = pd.DataFrame(vectorized.toarray(), columns=tfidf.get_feature_names_out())
print(vectorized_df)

tmp = pd.concat([data_processed, vectorized_df], axis=1).drop(columns=exclude_columns)
print(tmp.head())
print(get_score(tmp, data_processed['zestimate']))
data_processed = pd.concat([data_processed, vectorized_df], axis=1)
print(data_processed.head())

# Работа с геоданными

from sklearn.cluster import KMeans

print(len(data_processed['zipcode'].unique()))
kmeans = KMeans(n_clusters=15)
cluster = kmeans.fit_predict(data_processed[['latitude', 'longitude']])
print(cluster)

tmp = data_processed.copy()
tmp['cluster'] = cluster
print(tmp.head())
print(get_score(tmp.drop(columns=exclude_columns), data_processed['zestimate']))

data_processed = tmp.copy()

import reverse_geocoder as revgc

revgc.search((data_processed.iloc[10].latitude, data_processed.iloc[10].longitude))
