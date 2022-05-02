import os
import pandas as pd
from keras import backend as K
from keras.models import Model, Input
from keras.layers import Masking, TimeDistributed, multiply, Dense, LSTM, Permute, Reshape
from tensorflow.keras import optimizers, metrics, models
from lstm import attention_3d_block

'''Architecture notes

Multilabel prediction would predict joint labels, e.g., [1,1,0] or [1,0,0]
    - See Keras tutorial: https://keras.io/examples/nlp/multi_label_classification/
Consider predicting MI and sepsis, then passing output to death prediction
    - Multitask learning: https://datascience.stackexchange.com/questions/27498/multi-task-learning-in-keras
    `model = Model(inputs=x, outputs=[out1, out2, out3])`
    - Note the outputs need to have the same loss function
    - We could load in the base LSTMs and freeze the bottom layers, too!
'''

def get_mimic_multitask_matrices(df, targets, time_steps):
    # Pull matrix of data
    x_matrix = df[df.columns.drop(targets)].values.reshape(int(df.shape[0] / time_steps), time_steps, df.shape[1] - len(targets))
    y_matrix = df[targets].values.reshape(int(df.shape[0] / time_steps), time_steps, len(targets))
    print("Shape of X matrix:", x_matrix.shape)
    print("Shape of Y matrix:", y_matrix.shape)
    return x_matrix, y_matrix

'''Building on architecture used by Kaji et al. (2018)
INPUTS: 
X :: Matrix of size (number_samples, time_steps, number_feats)
'''
def get_multitask_shared_layers(X):
    # SHARED LAYERS
    _, time_steps, number_feats = X.shape
    input_layer = Input(shape=(time_steps, number_feats))
    shared_layers = attention_3d_block(input_layer, time_steps)
    shared_layers = Masking(mask_value=0, input_shape=(time_steps, number_feats))(shared_layers)
    shared_layers = LSTM(256, return_sequences=True)(shared_layers)
    return input_layer, shared_layers

if __name__=="__main__":
    # Initial checks required to run script
    assert os.path.isdir('models'), "Please create 'models/' directory before running"

    # Global params
    #targets = ['MI', 'Sepsis']
    targets = ['MI', 'Sepsis', 'DEATH']
    drop_cols = ['HADM_ID', 'SUBJECT_ID', 'HADMID_DAY', 'DOB', 'ADMITTIME'] +\
                ['hr_sepsis', 'respiratory rate_sepsis', 'wbc_sepsis', 'temperature f_sepsis', 'sepsis_points'] + \
                ['troponin', 'troponin_std', 'troponin_min', 'troponin_max'] +\
                ['DEATHTIME']
    time_steps = 14
    model_location = 'models/multitask_' + '_'.join([target.lower() for target in targets])

    '''
    # Get data and perform preprocessing
    train_df = pd.read_csv('data/train_padded.csv')
    train_df.drop(columns=drop_cols, inplace=True)
    train_df = train_df.fillna(0)
    X, y = get_mimic_multitask_matrices(train_df, targets, time_steps)
    y_dict = {target.lower(): y[:, :, [i]] for i, target in enumerate(targets)}

    # Train multi-task classifier
    batch_size = 128
    epochs = 10
    input_layer, shared_layers = get_multitask_shared_layers(X)

    # MI LAYERS
    mi_layers = Dense(128, activation='relu', name='mi_dense')(shared_layers)
    mi_pred = TimeDistributed(Dense(1, activation="sigmoid"), name='mi')(mi_layers)

    # SEPSIS LAYERS
    sepsis_layers = Dense(128, activation='relu', name='sepsis_dense')(shared_layers)
    sepsis_pred = TimeDistributed(Dense(1, activation="sigmoid"), name='sepsis')(sepsis_layers)

    # DEATH LAYERS
    death_layers = Dense(128, activation='relu', name='death_dense')(shared_layers)
    death_pred = TimeDistributed(Dense(1, activation="sigmoid"), name='death')(death_layers)

    # COMPILE MODEL WITH COMBINED OUTPUTS AND OUTCOME-SPECIFIC METRICS
    model = Model(inputs=input_layer, outputs=[mi_pred, sepsis_pred, death_pred])
    METRICS = {target.lower(): [metrics.BinaryAccuracy(name='accuracy'), metrics.AUC(name='auc')] for target in targets}
    RMS = optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08)
    model.compile(optimizer=RMS, loss=['binary_crossentropy'] * len(METRICS), metrics=METRICS)
    model.summary()

    # TRAIN MODEL
    history = model.fit(x=X, y=y_dict, batch_size=batch_size, epochs=epochs)
    model.save(model_location)
    print("Model saved:", model_location)

    '''
    # EVALUATE MODEL
    test_df = pd.read_csv('data/test_padded.csv')
    test_df.drop(columns=drop_cols, inplace=True)
    test_df = test_df.fillna(0)
    x_test, y_test = get_mimic_multitask_matrices(test_df, targets, time_steps)
    y_dict = {target.lower(): y_test[:, :, [i]] for i, target in enumerate(targets)}
    model = models.load_model(model_location)
    print("Evaluating test data using saved model loaded from:", model_location)
    model.evaluate(x_test, y_dict)
