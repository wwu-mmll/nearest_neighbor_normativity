import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from photonai import RegressionPipe, PipelineElement, Hyperpipe
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVR
from scipy.stats import spearmanr
from shared import BRAIN_MEASURES, ROOT_FOLDER, KNN_TRAIN_DATA


KNN_BRAIN_AGE_PATH = ROOT_FOLDER / 'knn_brainage'
KNN_BRAIN_AGE_BIAS_MODEL = KNN_BRAIN_AGE_PATH / 'linear_bias_corrector.pkl'
KNN_BRAIN_AGE_photonai_model = KNN_BRAIN_AGE_PATH / 'best_model.photonai'
KNN_BRAIN_AGE_SCALER = KNN_BRAIN_AGE_PATH


def train_brainage(train_features, train_targets):

    analysis = RegressionPipe('knn_train_data_brain_age',                              
                              outer_cv=None,
                              inner_cv=ShuffleSplit(n_splits=1, test_size=0.1),
                              optimizer='grid_search',
                              optimizer_params={},
                              metrics=['mean_absolute_error',
                                       'mean_squared_error',                                       
                                       'explained_variance'],
                              add_default_pipeline_elements=False,
                              random_seed=1337,
                              cache_folder='/home/rleenings/Projects/ContrastiveMDD/tmp_knn_brain_age_models/cache',
                              project_folder=KNN_BRAIN_AGE_PATH,                              
                              verbosity=2)

    
    analysis += PipelineElement('StandardScaler')
    analysis += PipelineElement('SVR', hyperparameters={
                                                        'C': [0.5, 1, 5, 10, 100, 500, 1000, 5000],
                                                        'gamma': ['scale', 'auto']
                                                        })
    
    analysis.fit(train_features, train_targets)

def load_brainage_model():
    return Hyperpipe.load_optimum_pipe(str(KNN_BRAIN_AGE_photonai_model))


def get_age_index(age_of_interest, array_of_interest, max_age, bin_size=1):
    start_age = max(age_of_interest - bin_size, 18)
    stop_age = min(age_of_interest + bin_size, max_age)
    older_than_min = array_of_interest >= start_age
    younger_than_max = array_of_interest < stop_age
    subgroup_index = older_than_min & younger_than_max
    return subgroup_index


def train_bias_corrector(correction_features, correction_targets):
    # todo: extract to function and load brain age model?
    # what about indices
    #  Bias correction
    # ------------------------------------
    # x = ay + b
    # x_corr = (x − b) /a
    prediction_pipe = load_brainage_model()
    predicted_brain_age = prediction_pipe.predict(correction_features)
    predicted_brain_age = predicted_brain_age.reshape((-1, 1))
    linear_bias_model = SVR(kernel='rbf', C=1000)
    linear_bias_model.fit(predicted_brain_age, correction_targets)

    # linear_bias_model = LinearRegression()
    # linear_bias_model.fit(predicted_brain_age, correction_targets)
    # a = linear_bias_model.coef_[0]
    # b = linear_bias_model.intercept_
    joblib.dump(linear_bias_model, KNN_BRAIN_AGE_BIAS_MODEL)


def apply_svm_brain_age(data_loader):
    # load features
    data_loader.load_data('knn_data')
    x = data_loader.data[BRAIN_MEASURES].values
    y = data_loader.data[data_loader.specifier.age_encoded]

    # load hyperpipe
    # CAT12BRAINAGE_photonai_model = ROOT_FOLDER / 'cat12_brainage' / 'best_model.photonai'
    prediction_pipe = load_brainage_model()
    predictions = prediction_pipe.predict(x)

    # apply bias correction
    linear_bias_correction = joblib.load(KNN_BRAIN_AGE_BIAS_MODEL)
    corrected_predictions = linear_bias_correction.predict(predictions.reshape((-1, 1)))

  

if __name__ == '__main__':
    data = pd.read_csv(KNN_TRAIN_DATA)    
    features = data[BRAIN_MEASURES].values
    target = data['age'].values

    training_index, correction_index = next(
        iter(ShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(features)))
    train_features, train_targets = features[training_index], target[training_index]
    correction_features, correction_targets = features[correction_index], target[correction_index]

    train_brainage(train_features, train_targets)
    # todo: best model needs to be copied to root folder
    train_bias_corrector(correction_features, correction_targets)

