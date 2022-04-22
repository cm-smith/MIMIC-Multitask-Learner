import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_mimic_data(loc="data/CHARTEVENTS_reduced_24_hour_blocks_plus_admissions_plus_patients_plus_scripts_plus_icds_plus_notes.csv"):
    print("Reading in MIMIC data at the HADMI_ID/HADMID_DAY-level")
    df = pd.read_csv(loc)
    assert sum(df.groupby(['HADM_ID', 'HADMID_DAY']).size().values) == df.shape[0], 'ERROR: Not at the patient-day level'
    print("Fetched data of shape:", df.shape)
    return df

def filter_mimic_day1(df):
    print("Filtering only on day 1 in the ICU")
    df.loc[:, 'icu_day'] = df.sort_values(['HADM_ID', 'HADMID_DAY']).groupby(['HADM_ID']).cumcount() + 1
    day1_df = df[df['icu_day'] == 1].drop('icu_day', axis=1)
    #day1_df = df.groupby('HADM_ID').filter(lambda x: x['icu_day'].max() <= 1.).first().reset_index()
    print("Baseline data shape:", day1_df.shape)
    assert day1_df['HADM_ID'].nunique() == day1_df.shape[0], "ERROR: Data not 1 unique person per row as expected"
    return day1_df

class MimicData:
    def __init__(self, df, impute_outliers=True):
        self.df = df.copy()
        self.drop_cols = ['HADM_ID', 'SUBJECT_ID', 'HADMID_DAY', 'DOB', 'ADMITTIME']
        self.target = None
        if impute_outliers:
            self.impute_outliers()

    def __check_target(self):
        assert self.target is not None, "Target not defined"

    def __check_train_test(self):
        assert self.train is not None, "Train/test do not exist; please run MimicData.split_train_test()"

    def __check_numeric_cols(self, df=None):
        if df is None: df = self.df
        non_numeric_cols = {col: typ for col, typ in zip(df.columns.tolist(), df.dtypes.tolist())
                            if not np.issubdtype(typ, np.number)}
        if len(non_numeric_cols):
            print("WARNING: Non-numeric columns identified:", non_numeric_cols)
        return

    def impute_outliers(self, impute_cols=['AGE'], impute_percentile=0.99):
        for col in impute_cols:
            assert col in self.df.columns, 'ERROR: Column "'+col+'" to trim does not exist'
            outlier_threshold = self.df[col].quantile(impute_percentile)
            impute_rows = self.df[col] >= outlier_threshold
            impute_median = self.df[col].quantile(0.5)
            self.df.loc[impute_rows, col] = impute_median
            print("Outliers (>%.1f%%) in '%s' above %.1f imputed with median: %.2f" %
                  (impute_percentile*100, col, outlier_threshold, impute_median))
        return

    def get_feats(self, df=None, verbose=False):
        if df is None: df = self.df
        drop_col_list = [col for col in self.drop_cols+[self.target] if col in df.columns]
        ignore_list = set(self.drop_cols) - set(drop_col_list)
        if len(ignore_list) > 0 and verbose:
            print("Some columns not found (ignored):", ignore_list)
        X = df.drop(drop_col_list, axis=1)
        self.__check_numeric_cols(X)
        if verbose:
            print("Dropping columns:", drop_col_list)
            print("Shape of X features:", X.shape)
        return X

    def get_train_feats(self):
        self.__check_train_test()
        X_train = self.get_feats(df=self.train, verbose=True)
        return X_train

    def get_train_target(self):
        self.__check_target()
        self.__check_train_test()
        y_train = self.train[self.target]
        return y_train

    def get_test_feats(self):
        self.__check_train_test()
        X_test = self.get_feats(df=self.test)
        return X_test

    def get_test_target(self):
        self.__check_target()
        self.__check_train_test()
        y_test = self.test[self.target]
        return y_test

    def split_train_test(self, test_split=0.3, random_state=0, verbose=True, **kwargs):
        if self.target is not None: stratify = self.df[[self.target]]
        else: stratify = None
        self.train, self.test = train_test_split(self.df, test_size=test_split, random_state=random_state,
                                                 stratify=stratify, **kwargs)
        if verbose:
            print("Splitting data with %.2f%% test split" % (test_split*100))
            print("Train data has shape:", self.train.shape)
            print("Test data shape:", self.test.shape)
            if stratify is not None:
                print("Stratified by target variable:", self.target)
                print("%d/%d events in train; %d/%d events in test" %
                      (self.train[self.train[self.target]==1].shape[0], self.train.shape[0],
                       self.test[self.test[self.target]==1].shape[0], self.test.shape[0]))
        return

class MimicDataMI(MimicData):
    def __init__(self, df):
        MimicData.__init__(self, df)
        #self.drop_cols + ['ct_angio', 'troponin', 'troponin_std', 'troponin_min', 'troponin_max', 'Infection',
        #                  'CKD', 'hr_sepsis', 'respiratory rate_sepsis', 'wbc_sepsis', 'temperature f_sepsis',
        #                  'sepsis_points', 'vancomycin', 'HADM_ID', 'SUBJECT_ID', 'YOB', 'ADMITYEAR', 'DOB', 'Sepsis',
        #                  'day_counts', 'HADMID_DAY']
        self.target = 'MI'
        self.__append_mi_outcome()

    def __append_mi_outcome(self):
        self.df.loc[:, self.target] = ((self.df['troponin'] > 0.4) & (self.df['CKD'] == 0)).apply(lambda x: int(x))
        print("Total MI events:")
        print(self.df[self.target].value_counts())
        self.drop_cols = self.drop_cols + ['troponin', 'troponin_std', 'troponin_min', 'troponin_max']

class MimicDataSepsis(MimicData):
    def __init__(self, df):
        MimicData.__init__(self, df)
        self.target = 'Sepsis'
        self.__append_sepsis_outcome()

    ''' White blood cell (WBC) critical values
    INPUTS: col = df['WBCs']
    '''
    def __wbc_crit(self, col):
        if (col > 12 or col < 4) and col != 0:
            return 1
        else:
            return 0

    ''' Temperature (F) outside normal range
    INPUTS: col = df['temperature (F)']
    '''
    def __temp_crit(self, col):
        if (col > 100.4 or col < 96.8) and col != 0:
            return 1
        else:
            return 0

    def __append_sepsis_outcome(self):
        self.df.loc[:, 'hr_sepsis'] = self.df['heart rate'].apply(lambda x: 1 if x > 90 else 0)
        self.df.loc[:, 'respiratory rate_sepsis'] = self.df['respiratory rate'].apply(lambda x: 1 if x > 20 else 0)
        self.df.loc[:, 'wbc_sepsis'] = self.df['WBCs'].apply(self.__wbc_crit)
        self.df.loc[:, 'temperature f_sepsis'] = self.df['temperature (F)'].apply(self.__temp_crit)
        self.df.loc[:, 'sepsis_points'] = (self.df['hr_sepsis'] + self.df['respiratory rate_sepsis']
                                      + self.df['wbc_sepsis'] + self.df['temperature f_sepsis'])
        self.df.loc[:, self.target] = ((self.df['sepsis_points'] >= 2) & (self.df['Infection'] == 1)).apply(lambda x: int(x))
        self.df.loc[:, self.target].fillna(0)
        print("Total sepsis events:")
        print(self.df.loc[:, self.target].value_counts())
        self.drop_cols = self.drop_cols + ['hr_sepsis', 'respiratory rate_sepsis', 'wbc_sepsis', 'temperature f_sepsis', 'sepsis_points']

''' Meant to deal with padding for an RNN when keras.preprocessing.pad_sequences fails
Refactored code from Deepak Kaji (2018)
https://github.com/deepak-kaji/mimic-lstm
'''
class PadSequences(object):
    def __init__(self):
        self.name = 'padder'

    ''' Takes a file path for the dataframe to operate on. lb is a lower bound to discard
        ub is an upper bound to truncate on. All entries are padded to their ubber bound
    '''
    def pad(self, df, lb, time_steps, pad_value=-100):
        self.uniques = pd.unique(df['HADM_ID'])
        df = df.groupby('HADM_ID').filter(lambda group: len(group) > lb).reset_index(drop=True)
        df = df.groupby('HADM_ID').apply(lambda group: group[0:time_steps]).reset_index(drop=True)
        df = df.groupby('HADM_ID').apply(lambda group: pd.concat([group, pd.DataFrame(pad_value*np.ones((time_steps-len(group), len(df.columns))), columns=df.columns)], axis=0)).reset_index(drop=True)
        return df

    ''' Performs Z Score Normalization for 3rd order tensors
        matrix should be (batchsize, time_steps, features)
        Padded time steps should be masked with np.nan
    '''
    def ZScoreNormalize(self, matrix):
        x_matrix = matrix[:,:,0:-1]
        y_matrix = matrix[:,:,-1]
        print(y_matrix.shape)
        y_matrix = y_matrix.reshape(y_matrix.shape[0],y_matrix.shape[1],1)
        means = np.nanmean(x_matrix, axis=(0,1))
        stds = np.nanstd(x_matrix, axis=(0,1))
        print(x_matrix.shape)
        print(means.shape)
        print(stds.shape)
        x_matrix = x_matrix-means
        print(x_matrix.shape)
        x_matrix = x_matrix / stds
        print(x_matrix.shape)
        print(y_matrix.shape)
        matrix = np.concatenate([x_matrix, y_matrix], axis=2)
        return matrix

    ''' Performs a NaN/pad-value insensiive MinMaxScaling
        When column maxs are zero, it ignores these columns for division
    '''
    def MinMaxScaler(self, matrix, pad_value=-100):
        bool_matrix = (matrix == pad_value)
        matrix[bool_matrix] = np.nan
        mins = np.nanmin(matrix, axis=0)
        maxs = np.nanmax(matrix, axis=0)
        matrix = np.divide(np.subtract(matrix, mins), np.subtract(maxs,mins), where=(np.nanmax(matrix,axis=0) != 0))
        matrix[bool_matrix] = pad_value
        return matrix
