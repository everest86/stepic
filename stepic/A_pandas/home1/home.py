import pandas as pd
from datetime import datetime

# Показать все столбцы
pd.set_option('display.max_columns', None)
# По желанию: увеличить ширину строки, чтобы столбцы не переносились
pd.set_option('display.width', 1000)

# загрузка данных
subscribers = pd.read_csv('subscribers.csv')
users = pd.read_csv('users.csv')
marketing = pd.read_csv('marketing_campaign.csv')

# количество строк и столбцов
subsRowsCount, subsColumnCount = subscribers.shape
print(f"subscribers: {subsRowsCount} row count, {subsColumnCount} column count")

# количество строк и столбцов
usRowsCount, usColumnCount = users.shape
print(f"users: {usRowsCount} row count, {usColumnCount} column count")

# количество строк и столбцов
markRowsCount, markColumnCount = marketing.shape
print(f"marketing_campaign: {markRowsCount} row count, {markColumnCount} column count")

print(subscribers)
print(users)
print(marketing)

# 1 объединение трех таблиц
mergedTable = subscribers.merge(marketing.merge(users, on='user_id', how='inner'), on='user_id', how='inner')
print(mergedTable.head(2))

# 2 типы данных
print(mergedTable.dtypes)
# статистика
print(mergedTable.describe())

# 3 эффективность маркетинговых каналов по платящим
moreEffectiveMarketingChannels = marketing[marketing['converted'] == True].groupby('marketing_channel')['variant'].count().sort_values(ascending=False)
print(moreEffectiveMarketingChannels)

# 4 Количество игроков в каждой возрастной группе
usersCountByAgeGroup=users.groupby(['age_group'])['age_group'].count()
print(usersCountByAgeGroup)

# 5 Самая ранняя дата подписки
minSubscriptionDate = subscribers['date_subscribed'].astype('datetime64[ns]').min()
print(minSubscriptionDate)

# 6 Портрет аудитории удержанных подписчиков
retainedUsers = subscribers[subscribers['is_retained']==True].merge(users, on='user_id', how='inner').merge(marketing, on='user_id', how='inner').groupby(['language_displayed','age_group'])['is_retained'].count()
print(retainedUsers)