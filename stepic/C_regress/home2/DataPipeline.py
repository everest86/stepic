from sklearn.base import BaseEstimator, TransformerMixin


class DataPipeline(BaseEstimator, TransformerMixin):
    """Подготовка исходных данных"""

    def __init__(self):
        """Параметры класса"""
        self.max_kms = None
        self.median_kms = None

    def fit(self, df, y=None):
        """Сохранение статистик"""

        # Расчет медиан
        self.max_kms = 230000
        self.median_kms = df['Kms_Driven'].median()

        return self

    def transform(self, df, y=None):
        """Трансформация данных"""

        df.loc[df['Kms_Driven'] < self.max_kms, 'Kms_Driven'] = self.median_kms

        return df
