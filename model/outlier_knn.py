import pickle
# import matplotlib
# matplotlib.use('TkAgg')  # Qt5Agg
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import pandas as pd
from scipy import stats
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from shared import PLOT_PATH, BRAIN_MEASURES




class ZDistributionScore:

    def __init__(self, name, distribution=None):
        self.name = name
        self.distribution = distribution if distribution is not None else stats.exponweib

    def __call__(self, density_values, fit):
        if fit is True:
            try:
                self.best_fit_params = self.distribution.fit(density_values)
            except RuntimeWarning as e:
                plt.figure()
                plt.hist(density_values)
                plt.show()
                print(e)
        distribution_data = self.distribution.pdf(density_values, *self.best_fit_params)

        if fit is True:
            self.distribution_max = np.max(distribution_data)
            turning_point = np.argmax(distribution_data == self.distribution_max)
            self.thresh_value = float(density_values[turning_point])

        distribution_score = distribution_data

        indexer = density_values < self.thresh_value
        distribution_score[indexer] = (self.distribution_max - distribution_score[indexer]) + self.distribution_max
        if fit is True:
            self.score_max = np.max(distribution_score)
        distribution_score = self.score_max - distribution_score

        if fit is True:            
            self.score_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(distribution_score.reshape((-1, 1)))            

        distribution_score = self.score_scaler.transform(distribution_score.reshape((-1, 1)))        
        z_scores = np.squeeze(distribution_score)
        return z_scores


class OutlierKNN:
    PREFERRED_DISTRIBUTION = stats.t
    BANDWIDTH = 0.05

    @staticmethod
    def gender_encoder_func(gender):
        # return 'female' if gender == 2 else 'male'
        return str(gender)

    def __init__(self, age, gender, save_models_to,
                 age_bin_size=1,
                 relative_neighbors=0.1,
                 max_neighbors=15):

        self.save_models_to = save_models_to

        self.age = age
        self.gender = gender
        self.gender_encoder = self.gender_encoder_func(self.gender)
        self.age_bin_size = age_bin_size

        self.relative_neighbors = relative_neighbors
        self.max_neighbors = max_neighbors
        self.num_neighbors = None

        self.median_distance = None
        self.median_density = None
        self.mad_density = None

        self.nn = None
        self.scaler = None        

        self.metadata = None

        self.plot_output_root = self.get_plot_path()
        if not self.plot_output_root.is_dir():
            self.plot_output_root.mkdir(parents=True, exist_ok=True)

    def get_plot_path(self):
        return PLOT_PATH.joinpath(f'train_knn/{self.gender_encoder}/{self.age_bin_size}')

    def norm_neighborhood(self, data_to_fit, plot=True):
        norm_dens = self.neighborhood(data_to_fit, fit=True)
        z_scores = self.z_scores(norm_dens, fit=True)
        if plot:
            self.distribution_plot(norm_dens, z_scores)

        # store metadata
        self.metadata = {'age': self.age, 'gender': self.gender, 'total_samples': data_to_fit.shape[0],
                         'num_neighbors': self.num_neighbors, 'median_distance': self.median_distance,
                         'time_of_training': str(datetime.now())}

        if "study_name" in data_to_fit.columns:
            for s in data_to_fit["study_name"].unique():
                num_of_samples_per_study = data_to_fit[data_to_fit["study_name"] == s].shape[0]
                self.metadata[f"num_samples_{s}"] = num_of_samples_per_study

        return norm_dens

    def neighborhood(self, input_data, fit=False):
        if fit:
            self.input_scaler = MinMaxScaler().fit(input_data)

        input_data = self.input_scaler.transform(input_data)

        if fit:
            self.train_data = input_data
        

        if fit is True:
            self.num_neighbors = np.ceil(self.relative_neighbors * input_data.shape[0]).astype(int)
            self.num_neighbors = min(self.max_neighbors, self.num_neighbors)            
            self.nn = NearestNeighbors(n_neighbors=self.num_neighbors).fit(input_data)

        local_distances, _ = self.nn.kneighbors(input_data)
   
        if fit is True:
            self.median_distance = np.median(local_distances)

        # todo (maybe):
        # use gaussian weighting of distances to neighbors? is this reasonable ????

        # for distance approach:
        local_distances = local_distances / self.median_distance

        # ------ SUM OF NEIGHBOR DISTANCES --------
        density_values = 1.0 / np.sum(local_distances, axis=1)

    
        if fit is True:
            self.std_scaler = StandardScaler().fit(density_values.reshape(-1, 1))
        normalized_density_values = self.std_scaler.transform(density_values.reshape(-1, 1))        
        
        return normalized_density_values

    def z_scores(self, density_values, fit=False):
        if fit:
            self.z_scorer = ZDistributionScore("density", distribution=self.PREFERRED_DISTRIBUTION)
        return self.z_scorer(density_values, fit)

    def distribution_plot(self, maybe_skewed_data, z_scores, save_path=None):

        plt.figure()
        plt.hist(maybe_skewed_data)
        # plt.show()

        plt.figure()
        ax = plt.subplot()
        plt.hist(maybe_skewed_data, bins=20, density=True, alpha=0.6, color='g')
        x = np.linspace(min(maybe_skewed_data), max(maybe_skewed_data), maybe_skewed_data.shape[0])
        ax.plot(x, self.PREFERRED_DISTRIBUTION.pdf(x, *self.z_scorer.best_fit_params), 'r')
        ax.set_xlim((-4, 4))
        ax2 = ax.twinx()
        ax2.scatter(maybe_skewed_data, z_scores)

        title = f'{self.gender_encoder} | {self.age} | n={maybe_skewed_data.shape[0]}'
        plt.title(title)
        save_to = self.plot_output_root.joinpath(f'{self.age}.png') if save_path is None else save_path
        plt.savefig(save_to)
        plt.close()
        # plt.show()

    def file_name_func(self):
        return f'{self.age}_{self.gender_encoder}.pkl'

    def save(self):
        with open(self.save_models_to.joinpath(self.file_name_func()), 'wb') as file:
            pickle.dump(self, file)

    def load(self):
        with open(self.save_models_to.joinpath(self.file_name_func()), 'rb') as file:
            loaded_instance = pickle.load(file)
            return loaded_instance

    @staticmethod
    def dataframe_data(X, kwargs):
        data = np.concatenate((X, kwargs["age"].reshape((-1, 1)), kwargs["sex"].reshape((-1, 1))), axis=1)
        df = pd.DataFrame(data=data, columns=BRAIN_MEASURES + ["age", "sex"])
        return df


class OutlierCurveKNN(OutlierKNN):
    PREFERRED_DISTRIBUTION = stats.t#
    BANDWIDTH = 0.7

    def file_name_func(self):
        return f'{self.age}_{self.gender_encoder}_curve.pkl'

    def get_plot_path(self):
        return PLOT_PATH.joinpath(f'train_knn/{self.gender_encoder}/{self.age_bin_size}_curves')
