from sklearn.base import BaseEstimator, TransformerMixin


class DataPipeline(BaseEstimator, TransformerMixin):
    """Подготовка исходных данных"""

    def __init__(self):
        """Параметры класса"""
        self.binColumns = []
        self.catColumns = []
        self.numColumns = []
        self.binCatColumns = []

    def fit(self, df, y=None):

        for column in df.columns:
            if (df[column].value_counts().shape[0] == 2):
                self.binColumns.append(column)
            elif (df[column].dtypes == 'object'):
                self.catColumns.append(column)
            else:
                self.numColumns.append(column)

        self.binCatColumns = self.catColumns + self.binColumns
        return self

    def transform(self, df, y=None):

        if 'Loan_ID' in df.columns:
            df.drop(['Loan_ID'], axis=1, inplace=True)

        return df
