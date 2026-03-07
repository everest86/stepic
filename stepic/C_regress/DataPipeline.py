from sklearn.base import BaseEstimator, TransformerMixin


class DataPipeline(BaseEstimator, TransformerMixin):
    """Подготовка исходных данных"""

    def __init__(self):
        """Параметры класса"""
        self.longitude_max = None
        self.latitude_min = None
        self.latitude_max = None

        self.latitude_median = None
        self.longitude_median = None

    def fit(self, df, y=None):
        """Сохранение статистик"""

        # Расчет медиан
        self.longitude_max = 50
        self.latitude_min = -90
        self.latitude_max = 90

        if 'latitude' in df.columns:
            self.latitude_median = df['latitude'].median()
        if 'longitude' in df.columns:
            self.longitude_median = df['longitude'].median()

        return self

    def transform(self, df, y=None):
        """Трансформация данных"""

        if 'longitude' in df.columns:
            df.loc[df['longitude'] > self.longitude_max, 'longitude'] = self.longitude_median
        if 'latitude' in df.columns:
            df.loc[(df['latitude'] <= self.latitude_min) | (
                        df['latitude'] > self.latitude_max), 'latitude'] = self.latitude_median

        if 'index' in df.columns:
            df.drop(columns='index', inplace=True)
        if 'id' in df.columns:
            df.drop(columns='id', inplace=True)
        if 'ocean_proximity' in df.columns:
            df.drop(columns='ocean_proximity', inplace=True)

        return df