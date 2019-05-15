import unittest
import prep.train_val_test_split as train_val_test_split
import pandas as pd
import os


class TestTrainValTestSplit(unittest.TestCase):
    def test_train_val_test_split(self):
        root_train = 'data/train/'
        root_val = 'data/val/'
        root_test = 'data/test/'

        # without saving csv
        hour_train_x, hour_train_y, hour_val_x, hour_val_y, hour_test_x, hour_test_y = \
            train_val_test_split.train_val_test_split(False,
                                                      path_read_csv_hour='data/raw/hour.csv',
                                                      root_train=root_train,
                                                      root_val=root_val,
                                                      root_test=root_test
                                                      )
        # data type
        for data in (hour_train_x, hour_train_y, hour_val_x, hour_val_y, hour_test_x, hour_test_y):
            self.assertIsInstance(data, pd.DataFrame, 'Wrong data type.')

        # check for overlap
        overlap_train_val = len(hour_train_x.merge(hour_val_x, on='dteday'))
        overlap_train_test = len(hour_train_x.merge(hour_test_x, on='dteday'))
        overlap_val_test = len(hour_test_x.merge(hour_val_x, on='dteday'))
        self.assertEqual(overlap_train_val, 0, 'Overlap between training and validation data.')
        self.assertEqual(overlap_train_test, 0, 'Overlap between training and test data.')
        self.assertEqual(overlap_val_test, 0, 'Overlap between validation and test data.')

        # check if date of test/val data is past training data
        self.assertTrue(hour_train_x['dteday'].max()<hour_val_x['dteday'].min(),
                        'validation data is from earlier than training data.')
        self.assertTrue(hour_train_x['dteday'].max() < hour_test_x['dteday'].min(),
                        'validation data is from earlier than training data.')

        # check if csv files are saved
        paths =[(root_train + 'hour_x.csv'),
                (root_train + 'hour_y.csv'),
                (root_val + 'hour_x.csv'),
                (root_val + 'hour_y.csv'),
                (root_test + 'hour_x.csv'),
                (root_test + 'hour_y.csv')
                ]
        for path in paths:
            if os.path.isfile(path):
                os.remove(path)

        train_val_test_split.train_val_test_split(True,
                                                  path_read_csv_hour='data/raw/hour.csv',
                                                  root_train=root_train,
                                                  root_val=root_val,
                                                  root_test=root_test
                                                  )
        for path in paths:
            self.assertTrue(os.path.isfile(path), 'File not saved: '+path)
            self.assertIsInstance(pd.read_csv(path), pd.DataFrame, 'Cannot read format of file: '+path)
