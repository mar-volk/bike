import pandas as pd


def train_val_test_split(save_csv=True, *,
                         path_read_csv_hour='../../data/raw/hour.csv',
                         root_train='../../data/train/',
                         root_val='../../data/val/',
                         root_test='../../data/test/'
                         ):
    # read raw data
    parse_dates = ['dteday']
    hour = pd.read_csv(path_read_csv_hour, parse_dates=parse_dates)

    # idea of split:
    # 2011 --> train
    # 2012, even day of year --> validation
    # 2012, uneven day of year --> test

    limit_day = pd.to_datetime('2012-01-01')

    hour_train = hour.loc[hour['dteday'] < limit_day]
    hour_train_x = hour_train.drop(columns=['casual', 'registered', 'cnt', 'instant'])
    hour_train_y = hour_train[['cnt']]

    hour_val = hour.loc[(hour['dteday'] >= limit_day) & (hour['dteday'].apply(lambda z: z.dayofyear) % 2 == 0)]
    hour_val = hour_val.reset_index(drop=True)
    hour_val_x = hour_val.drop(columns=['casual', 'registered', 'cnt', 'instant'])
    hour_val_y = hour_val[['cnt']]

    hour_test = hour.loc[(hour['dteday'] >= limit_day) & (hour['dteday'].apply(lambda z: z.dayofyear) % 2 == 1)]
    hour_test = hour_test.reset_index(drop=True)
    hour_test_x = hour_test.drop(columns=['casual', 'registered', 'cnt', 'instant'])
    hour_test_y = hour_test[['cnt']]

    # Check for overlap between train, val and test
    overlap_train_val = len(hour_train_x.merge(hour_val_x, on='dteday'))
    overlap_train_test = len(hour_train_x.merge(hour_test_x, on='dteday'))
    overlap_val_test = len(hour_test_x.merge(hour_val_x, on='dteday'))
    assert overlap_train_val == 0, 'Overlap between training and validation data.'
    assert overlap_train_test == 0, 'Overlap between training and test data.'
    assert overlap_val_test == 0, 'Overlap between validation and test data.'

    # save data
    if save_csv:
        hour_train_x.to_csv(root_train + 'hour_x.csv', index=False)
        hour_train_y.to_csv(root_train + 'hour_y.csv', index=False)

        hour_val_x.to_csv(root_val + 'hour_x.csv', index=False)
        hour_val_y.to_csv(root_val + 'hour_y.csv', index=False)

        hour_test_x.to_csv(root_test + 'hour_x.csv', index=False)
        hour_test_y.to_csv(root_test + 'hour_y.csv', index=False)

    return hour_train_x, hour_train_y, hour_val_x, hour_val_y, hour_test_x, hour_test_y


if __name__ == '__main__':
    train_val_test_split()


