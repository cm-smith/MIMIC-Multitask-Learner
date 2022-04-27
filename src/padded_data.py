import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import get_mimic_data

''' White blood cell (WBC) critical values
INPUTS: col = df['WBCs']
'''
def __wbc_crit(col):
    if (col > 12 or col < 4) and col != 0:
        return 1
    else:
        return 0

''' Temperature (F) outside normal range
INPUTS: col = df['temperature (F)']
'''
def __temp_crit(col):
    if (col > 100.4 or col < 96.8) and col != 0:
        return 1
    else:
        return 0

'''Add target columns to given dataframe (df)
'''
def add_target_cols(df):
    print("Adding targets: 'MI', 'Sepsis'")
    df.loc[:, 'MI'] = ((df['troponin'] > 0.4) & (df['CKD'] == 0)).apply(lambda x: int(x))
    df.loc[:, 'hr_sepsis'] = df['heart rate'].apply(lambda x: 1 if x > 90 else 0)
    df.loc[:, 'respiratory rate_sepsis'] = df['respiratory rate'].apply(lambda x: 1 if x > 20 else 0)
    df.loc[:, 'wbc_sepsis'] = df['WBCs'].apply(__wbc_crit)
    df.loc[:, 'temperature f_sepsis'] = df['temperature (F)'].apply(__temp_crit)
    df.loc[:, 'sepsis_points'] = (df['hr_sepsis'] + df['respiratory rate_sepsis'] + df['wbc_sepsis'] + df['temperature f_sepsis'])
    df.loc[:, 'Sepsis'] = ((df['sepsis_points'] >= 2) & (df['Infection'] == 1)).apply(lambda x: int(x))
    df.loc[:, 'Sepsis'].fillna(0)
    return df

'''Drop patients with less than "min_days" in the ICU
'''
def filter_days(df, min_days):
    print("Filtering out patients that were in the ICU for less than %d days" % (min_days))
    new_df = df.groupby('HADM_ID').filter(lambda group: len(group) >= min_days).reset_index(drop=True)
    return new_df

'''Pad given dataframe (df) with specified pad values up to a maximum length
'''
def pad_df(df, max_len, pad_value=0):
    print("Padding sequences of length %d with value %d" % (max_len, pad_value))
    # Restrict to maximum length
    df = df.groupby('HADM_ID').apply(lambda group: group[0:max_len]).reset_index(drop=True)
    # Pad additional data at the patient-level
    df = df.groupby('HADM_ID').apply(lambda group: pd.concat(
        [group,
         pd.DataFrame(pad_value * np.ones((max_len - len(group), len(df.columns))), columns=df.columns)], axis=0)
                                             ).reset_index(drop=True)
    print("Shape of data after padding:", df.shape)
    return df

'''For values beyond the specified percentile threshold, impute with the median value
'''
def impute_outliers(df, impute_cols=['AGE'], impute_percentile=0.99):
    for col in impute_cols:
        assert col in df.columns, 'ERROR: Column "'+col+'" to trim does not exist'
        outlier_threshold = df[col].quantile(impute_percentile)
        impute_rows = df[col] >= outlier_threshold
        impute_median = df[col].quantile(0.5)
        df.loc[impute_rows, col] = impute_median
        print("Outliers (>%.1f%%) in '%s' above %.1f imputed with median: %.2f" %
              (impute_percentile*100, col, outlier_threshold, impute_median))
    return df

if __name__=="__main__":
    # Global params
    min_icu_days = 2      # Patients much have been in the ICU for at least this many days
    time_steps = 14       # Maximum length of ICU days in sequence passed to LSTM
    random_state = 1234   # Randomly splitting data consistently
    test_split = 0.3
    validation_split = 0.1

    # Fetch data and preprocess
    df = get_mimic_data()
    df = filter_days(df, min_days=min_icu_days)
    df = impute_outliers(df, impute_cols=['AGE'], impute_percentile=0.99)
    df = add_target_cols(df)
    assert 'MI' in df.columns, "ERROR: Unsuccessful attempt to add target columns"

    # Split data
    patients = pd.unique(df['HADM_ID'])
    train_pt, test_pt = train_test_split(patients, test_size=test_split, random_state=random_state)
    train_pt, val_pt = train_test_split(train_pt, test_size=validation_split, random_state=random_state)

    # Save data splits to CSV files in 'data/' directory
    patient_sets = {'train': train_pt, 'test': test_pt, 'validation': val_pt}
    for patient_label in patient_sets:
        print("Padding data for %s split" % patient_label)
        split_df = df[df['HADM_ID'].isin(patient_sets[patient_label])]
        split_df = pad_df(split_df, max_len=time_steps, pad_value=0)
        split_df = split_df.replace({0: np.nan})
        split_df = split_df.round(2)
        csv_file = 'data/'+patient_label+'_padded.csv'
        split_df.to_csv(csv_file, index=False)
        print("Data written to file:", csv_file)
