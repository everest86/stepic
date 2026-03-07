class DataPipeline:
    """Подготовка исходных данных"""

    def __init__(self):
        """Параметры класса"""
        self.longitude_max = None
        self.latitude_min = None
        self.latitude_max = None

        self.latitude_median = None
        self.longitude_median = None

    def fit(self, df):
        """Сохранение статистик"""

        # Расчет медиан
        self.longitude_max = 50
        self.latitude_min = -90
        self.latitude_max = 90

        self.latitude_median = df['latitude'].median()
        self.longitude_median = df['longitude'].median()

    def transform(self, df):
        """Трансформация данных"""

        df.loc[df['longitude'] > self.longitude_max, 'longitude'] = self.longitude_median
        df.loc[(df['latitude'] <= self.latitude_min) | (
                df['latitude'] > self.latitude_max), 'latitude'] = self.latitude_median

        # удаляем ненужные колонки
        if 'index' in df.columns:
            df.drop(columns='index', inplace=True)
        if 'id' in df.columns:
            df.drop(columns='id', inplace=True)
        if 'ocean_proximity' in df.columns:
            df.drop(columns='ocean_proximity', inplace=True)

        return df
