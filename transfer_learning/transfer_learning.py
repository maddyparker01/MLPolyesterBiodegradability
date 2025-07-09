import keras
import numpy as np
import tensorflow as tf
from keras import layers, models
from numpy import random
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.pipeline import Pipeline

from featurisation.feature_transformers import get_data, get_fransen_data, getFingerprints

print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU Available:", tf.config.list_physical_devices('GPU'))

random.seed(42)
tf.random.set_seed(42)

def build_model(input_shape, units):
    base_model = models.Sequential()
    base_model.add(layers.Dense(units=units, activation='relu', input_shape=(input_shape,)))
    base_model.add(layers.Dense(units=int(units/2), activation='relu'))
    base_model.add(layers.Dense(1, activation='sigmoid'))
    base_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

    return base_model


def build_transfer_model(base_model):
    fold_base_model = keras.models.clone_model(base_model)
    fold_base_model.set_weights(base_model.get_weights())

    transfer_model = models.Sequential([
        fold_base_model,
        layers.Dense(1, activation='sigmoid')
    ])
    transfer_model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
    return transfer_model


# GET TARGET DATA ----------------------------------------------------
mols_new, y = get_data()

# preprocess
pipeline = Pipeline([
    ('fingerprint', getFingerprints(fpSize=2048, maxPath=6)),
    ('threshold', VarianceThreshold())
])

X = pipeline.fit_transform(mols_new)

# GET SOURCE DATA ----------------------------------------------------
mols, base_y = get_fransen_data()

base_X = pipeline.transform(mols)


# START MODEL ----------------------------------------------------
# define CV
kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

# define search grid
hyperparams = {'units': [32, 64, 128, 256],
               'epochs': [10, 20, 30, 40, 50]}

# generate random configs
hyperparam_configs = list(ParameterSampler(hyperparams, n_iter=5, random_state=42))

# init full result lists
outer_testscores = []

for train_idx, test_idx in kf.split(X, y):  # 5CV FOR OUTER FOLD
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # track hyperparameter performance
    param_testscores = []

    for idx, config in enumerate(hyperparam_configs):  # iterate over hyperparams
        units = config['units']
        epochs = config['epochs']

        #train base model
        base_model = build_model(base_X.shape[1], units)
        base_model.fit(base_X, base_y, epochs=epochs, batch_size=32, verbose=0)
        base_model.pop()
        base_model.trainable = False

        # init inner fold results
        inner_testscores = []

        for train_index, test_index in kf.split(X_train, y_train):  # 5CV for inner fold
            innerX_train, innerX_test = X_train[train_index], X_train[test_index]
            innery_train, innery_test = y_train[train_index], y_train[test_index]

            # retrain output layer
            transfer_model = build_transfer_model(base_model)
            transfer_model.fit(innerX_train, innery_train, epochs=10, batch_size=6, verbose=0, validation_split=0.1)

            # fine tune
            transfer_model.trainable = True
            transfer_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.00001),  # Low learning rate for fine-tuning
                loss='binary_crossentropy',
                metrics=['accuracy'],
            )
            transfer_model.fit(innerX_train, innery_train, epochs=10, batch_size=6, verbose=0, validation_split=0.1)

            # evaluate
            accuracy = transfer_model.evaluate(innerX_test, innery_test)[1]
            inner_testscores.append(accuracy)

        param_testscores.append(np.mean(inner_testscores))

    # FINAL MODAL: refit with best hyperparams
    param_idx = param_testscores.index(max(param_testscores))
    best_config = hyperparam_configs[param_idx]

    #train base model
    base_model = build_model(base_X.shape[1], best_config['units'])
    base_model.fit(base_X, base_y, epochs=best_config['epochs'], batch_size=32, verbose=0)
    base_model.pop()
    base_model.trainable = False

    # transfer
    transfer_model = build_transfer_model(base_model)
    transfer_model.fit(X_train, y_train, epochs=10, batch_size=6, verbose=0)

    # fine tune
    transfer_model.trainable = True
    transfer_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00001),  # Low learning rate for fine-tuning
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    transfer_model.fit(X_train, y_train, epochs=10, batch_size=6, verbose=0, validation_split=0.1)

    # evaluate
    accuracy = transfer_model.evaluate(X_test, y_test)[1]
    outer_testscores.append(accuracy)

final_accuracy = np.mean(outer_testscores)
print(final_accuracy)
