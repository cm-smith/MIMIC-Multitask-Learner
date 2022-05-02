import pandas as pd
import os

from keras import backend as K
from keras.models import Model, Input
from keras.layers import Masking, TimeDistributed, multiply, Dense, LSTM, Permute, Reshape
from tensorflow.keras import optimizers, metrics, models

'''Refactored code from Deepak Kaji (2018)
https://github.com/deepak-kaji/mimic-lstm
'''
# inputs.shape = (batch_size, time_steps, input_dim)
def attention_3d_block(inputs, TIME_STEPS):
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul

def get_mimic_lstm_matrices(df, target, time_steps):
    # Pull matrix of data
    x_matrix = df[df.columns.drop(target)].values.reshape(int(df.shape[0] / time_steps), time_steps, df.shape[1] - 1)
    y_matrix = df[[target]].values.reshape(int(df.shape[0] / time_steps), time_steps, 1)
    print("Shape of X matrix:", x_matrix.shape)
    print("Shape of Y matrix:", y_matrix.shape)
    return x_matrix, y_matrix

'''Default architecture as used from the paper by Kaji et al. (2018)
INPUTS: 
X :: Matrix of size (number_samples, time_steps, number_feats)
'''
def train_mimic_lstm(X, y, batch_size=20, epochs=10, verbose=True):
    # Compile model
    _, time_steps, number_feats = X.shape
    input_layer = Input(shape=(time_steps, number_feats))
    x = attention_3d_block(input_layer, time_steps)
    x = Masking(mask_value=0, input_shape=(time_steps, number_feats))(x)
    x = LSTM(256, return_sequences=True)(x)
    preds = TimeDistributed(Dense(1, activation="sigmoid"))(x)
    model = Model(inputs=input_layer, outputs=preds)
    METRICS = [
        metrics.BinaryAccuracy(name='accuracy'),
        metrics.AUC(name='auc'),
    ]
    RMS = optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08)
    model.compile(optimizer=RMS, loss='binary_crossentropy', metrics=METRICS)

    if verbose:
        print("~~~ PRINTING MODEL SUMMARY ~~~")
        model.summary()

    # Train model
    history = model.fit(x=X, y=y, batch_size=batch_size, epochs=epochs)
    return model, history

if __name__=="__main__":
    # Initial checks required to run script
    assert os.path.isdir('models'), "Please create 'models/' directory before running"

    # Global params
    drop_cols = ['HADM_ID', 'SUBJECT_ID', 'HADMID_DAY', 'DOB', 'ADMITTIME']
    time_steps = 14

    # Model-specific params (choose one of these)
    target = 'DEATH'
    drop_cols = drop_cols + ['DEATHTIME']
    model_location = 'models/'+target.lower()+'_lstm'

    '''
    target = 'MI'
    drop_cols = drop_cols + ['troponin', 'troponin_std', 'troponin_min', 'troponin_max']
    model_location = 'models/'+target.lower()+'_lstm'
    
    target = 'Sepsis'
    drop_cols = drop_cols + ['hr_sepsis', 'respiratory rate_sepsis', 'wbc_sepsis', 'temperature f_sepsis', 'sepsis_points']
    model_location = 'models/'+target.lower()+'_lstm'
    '''

    '''
    # Train LSTM binary classifier
    train_df = pd.read_csv('data/train_padded.csv')
    train_df.drop(columns=drop_cols, inplace=True)
    train_df = train_df.fillna(0)
    x_train, y_train = get_mimic_lstm_matrices(train_df, target, time_steps)
    model, history = train_mimic_lstm(x_train, y_train, batch_size=20, epochs=10)
    model.save(model_location)
    print("Model saved:", model_location)

    '''
    # Evaluate LSTM on test data
    test_df = pd.read_csv('data/test_padded.csv')
    test_df.drop(columns=drop_cols, inplace=True)
    test_df = test_df.fillna(0)
    x_test, y_test = get_mimic_lstm_matrices(test_df, target, time_steps)
    model = models.load_model(model_location)
    print("Evaluating saved model on test data:")
    model.evaluate(x_test, y_test)
