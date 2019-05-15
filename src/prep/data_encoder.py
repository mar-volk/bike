import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class DataEncoder:
    def __init__(self):
        self.numeric_features = ['dteday', 'temp', 'atemp', 'hum', 'windspeed']
        self.categorical_features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
        self.numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                                   ('scaler', StandardScaler())])
        self.categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                                       ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        self.preprocessor = ColumnTransformer(transformers=[
            ('num', self.numeric_transformer, self.numeric_features),
            ('cat', self.categorical_transformer, self.categorical_features)])

    def _date_to_numeric(self, hour_x):
        hour_x_2 = hour_x.copy()
        hour_x_2['dteday'] = (hour_x['dteday'] - pd.to_datetime('2011-01-01')) / pd.Timedelta('1 days')
        return hour_x_2

    def fit(self, hour_x):
        hour_x_2 = self._date_to_numeric(hour_x)
        self.preprocessor.fit(hour_x_2)

    def transform(self, hour_x):
        hour_x_2 = self._date_to_numeric(hour_x)
        x = self.preprocessor.transform(hour_x_2).todense()
        return x


