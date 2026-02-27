import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 1. Загрузить, посмотреть, определить количество строк и объединить 3 датасета: marketing_campaign.csv, users.csv и subscribers.csv
subscribersDf = pd.read_csv('csv/subscribers.csv')
usersDf = pd.read_csv('csv/users.csv')
marketingDf = pd.read_csv('csv/marketing_campaign.csv')
mergedDf = usersDf.merge(marketingDf.merge(subscribersDf, how='inner', left_on='user_id', right_on='user_id'),
                         how='inner', left_on='user_id', right_on='user_id')
print(mergedDf.head(1))
rows, columns = mergedDf.shape
print(rows, columns)  # количество строк и колонок

# 2. Определить типы и статистики колонок
print(mergedDf.dtypes)  # типы колонок
print(mergedDf.describe())  # основные статистики

# 3. эффективность маркетинговых каналов по платящим
print(marketingDf[marketingDf['converted'] == True]['marketing_channel'].value_counts())

# 4. Определить количество игроков в каждой возрастной группе
print(usersDf['age_group'].value_counts())

# 5. Определить самую раннюю дату подписки на сервис
print(subscribersDf['date_subscribed'].astype('datetime64[ns]').min())

# 6. Определить портрет аудитории удержанных подписчиков (по возрасту и языку)
print(mergedDf[mergedDf['is_retained'] == True][['age_group', 'language_displayed']].value_counts())
