import matplotlib
# matplotlib.use('TkAgg')

import tqdm
import numpy as np
import pandas as pd
from scipy.integrate import quad
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from shared import DataLoader, BRAIN_MEASURES, KNN_MODEL_FOLDER
from .outlier_knn import OutlierKNN, OutlierCurveKNN
from pathlib import Path



class KNNProjection(BaseEstimator):
    z_score_column = "z_score"
    best_age_fit_column = "best_age_fit"
    age_fit_z_score = 'age_fit_z_score'
    density_column = "density_score"
    shift_column = "shift"
    abs_shift_column = f"abs_{shift_column}"
    age_specific_shift_column = f"age_specific_{shift_column}"
    pca_z_column = "pca_z"

    def __init__(self,
                 model_path: Path = None,
                 data_loader: DataLoader = None,
                 load_knn_train_data: bool = False,
                 norm_group: str = 'gender'):

        assert norm_group in ["age", "gender"]
        self.data_loader = data_loader
        self.load_knn_train_data = load_knn_train_data
        self.norm_group = norm_group
        self.model_path = model_path if model_path else KNN_MODEL_FOLDER
        if not self.model_path.is_dir():
            self.model_path.mkdir(parents=True, exist_ok=True)

        self.noise_thresh = {}

        self.train_mode = False
        self.plot = False

        # for photonai
        self.needs_covariates = True
        # self.needs_y = True

    def prepare_data_loader(self, X=None, kwarg_dict=None):
        if self.load_knn_train_data:
            self.data_loader = DataLoader.load_knn_train_data()
            self.data_loader.load_data(data_spec="raw_data", apply_filter=True)            
        elif self.data_loader is None:
            new_df = OutlierKNN.dataframe_data(X, kwarg_dict)
            self.data_loader = DataLoader.create(new_df, "knn_train_sklearn")

    def fit(self, X=None, y=None, **kwargs):

        assert("age" in kwargs)
        assert("sex" in kwargs)

        self.prepare_data_loader(X, kwargs)
        self.train_models()
        return self

    def transform(self, X=None, y=None, **kwargs):
        assert "age" in kwargs
        assert "sex" in kwargs
        self.data_loader = DataLoader.create(OutlierKNN.dataframe_data(X, kwargs), "knn_sklearn_project")
        self.project()
        z_scores = self.data_loader.data[self.z_score_column].values.reshape((-1, 1)).astype(float)
        best_age_z_score = self.data_loader.data[self.age_fit_z_score].values.reshape((-1, 1)).astype(float)
        shift_column = self.data_loader.data[self.age_specific_shift_column].values.reshape((-1, 1)).astype(float)
        X = np.concatenate((X, z_scores, best_age_z_score, shift_column), axis=1)        
        return X, kwargs

    def project(self):
        self.calculate_z()
        self.calculate_shift()

    def train(self):
        self.train_mode = True
        self.run()

    def run(self):

        self.plot = True
        self.prepare_data_loader()

        if self.train_mode:
            self.train_models()

        self.data_loader.load_data(data_spec="raw_data")
        self.calculate_z()
        self.data_loader.overwrite_csv("knn_data")

        self.data_loader.load_data(data_spec="knn_data")
        self.norm_curves()        
        self.calculate_shift()
        self.plot_age_range()        
        self.data_loader.overwrite_csv("knn_data")

        self.inspect()

    def train_models(self):
        
        reasonable_size = 75        
        max_bin_size = 1000         

        train_info_records = []
        self.data_loader.add_zero_columns(["density"])

        if not self.train_mode:
            pb = tqdm.tqdm(desc="Training KNN", total=(len(self.data_loader.age_range) * len(self.data_loader.genders)) + 2)
        for gender in self.data_loader.gender_iterator():

            gender_data = self.data_loader.get_gender_data(gender)
            if self.train_mode:
                print(f'{gender}: n={gender_data.shape[0]}')
            

            for current_age in self.data_loader.age_iterator():
                train_index = self.data_loader.get_age_gender_index_for_train(current_age, gender)
                age_data = self.data_loader.data[train_index]
                number_of_age_samples = age_data.shape[0]

                if number_of_age_samples > reasonable_size:
                    if self.train_mode:
                        print(f"training {number_of_age_samples:04d} items of "
                              f"age {current_age} with gender {gender} | train_knn")

                    if number_of_age_samples > max_bin_size:
                        permut_selector = np.random.permutation(np.arange(0, number_of_age_samples))
                        age_data = age_data.iloc[permut_selector[:max_bin_size]]                    

                    bag_obj = OutlierKNN(current_age, gender, self.model_path)
                    dens_values = bag_obj.norm_neighborhood(age_data[BRAIN_MEASURES], plot=self.plot)
                    train_info_records.append(bag_obj.metadata)
                    bag_obj.save()
                    
                    self.data_loader.data.loc[train_index, "density"] = dens_values                    

                    if not self.train_mode:
                        pb.update(1)

            if self.train_mode:
                metadata_df = pd.DataFrame.from_records(train_info_records)
                metadata_df.to_csv(self.data_loader.specifier.plot_folder.joinpath('metadata.csv'))

        # todo: update to only one z-scorer for both genders
        both_gender_indices = (self.data_loader.get_gender_index(1) | self.data_loader.get_gender_index(2))
        is_in_age_range = (self.data_loader.data[self.data_loader.specifier.age_encoded] > self.data_loader.lowest_valid_age)
        both_gender_indices = both_gender_indices & is_in_age_range
        for gender in self.data_loader.gender_iterator():
            gender_obj = OutlierKNN(0, gender, self.model_path)                        
            gender_density = self.data_loader.data.loc[both_gender_indices, "density"].values
            gender_z_scores = gender_obj.z_scores(gender_density, fit=True)
            self.noise_thresh[gender] = np.percentile(np.abs(gender_z_scores), 10)            
            if self.plot:
                gender_obj.distribution_plot(gender_density, gender_z_scores)
            gender_obj.save()
            if not self.train_mode:
                pb.update(1)

        if not self.train_mode:
            pb.close()

    def get_plot_path(self):
        if self.data_loader is None:
            raise ValueError("Cannot plot: There are no specifications because data_loader is None.")
        plot_path = self.data_loader.specifier.plot_folder
        plot_path.mkdir(parents=True, exist_ok=True)
        return plot_path

    def calculate_z(self):

        columns_to_add = [self.z_score_column, self.density_column]
        all_columns = [f'{alter}_{c_to_add}' for alter in self.data_loader.age_range for c_to_add in columns_to_add]
        all_columns += columns_to_add
        self.data_loader.add_zero_columns(all_columns)

        if not self.train_mode:
            pb = tqdm.tqdm(desc="Scoring data", total=(len(self.data_loader.genders) * len(self.data_loader.knn_range)))
        for gender in self.data_loader.gender_iterator():

            gender_BAG = OutlierKNN(0, gender, self.model_path).load()
            gender_index = self.data_loader.get_gender_index(gender)
            data_to_score = self.data_loader.get_brain_measures(gender_index)

            if data_to_score.shape[0] == 0:
                continue

            for age in self.data_loader.knn_age_iterator():

                alters_BAG = OutlierKNN(age, gender, self.model_path).load()
                z_scorer = alters_BAG if self.norm_group == 'age' else gender_BAG

                age_gender_index = self.data_loader.get_age_gender_index(age, gender)
                if self.train_mode:
                    print(f'scoring {np.sum(age_gender_index):04d} items of age {age} '
                          f'with gender {gender} | {self.data_loader.specifier.name}')

                if np.any(age_gender_index) and np.sum(age_gender_index) > 0:
                    specific_data_to_score = self.data_loader.get_brain_measures(age_gender_index)

                    alters_density = alters_BAG.neighborhood(specific_data_to_score, fit=False)
                    self.data_loader.data.loc[age_gender_index, self.density_column] = alters_density
                    self.data_loader.data.loc[age_gender_index, self.z_score_column] = z_scorer.z_scores(alters_density)

                gender_density = alters_BAG.neighborhood(data_to_score, fit=False)
                self.data_loader.data.loc[gender_index, f"{age}_{self.z_score_column}"] = z_scorer.z_scores(gender_density)

                if not self.train_mode:
                    pb.update(1)

            distribution_data = self.data_loader.data.loc[gender_index, self.density_column]
            G_z_score_data = self.data_loader.data.loc[gender_index, self.z_score_column]
            assert np.array_equal(distribution_data.isna(), G_z_score_data.isna())

            if self.plot:
                save_to = self.get_plot_path().joinpath(f"{self.data_loader.specifier.name}_{gender}.png")
                gender_BAG.distribution_plot(distribution_data.dropna(), G_z_score_data.dropna(), save_path=save_to)
        if not self.train_mode:
            pb.close()

    def norm_curves(self):

        columns_to_add = ["curve_density", "curve_z_age", "curve_z_total"]
        self.data_loader.add_zero_columns(columns_to_add)

        pb = tqdm.tqdm(desc=f"Norming curves per age", total=len(self.data_loader.knn_range))
        for gender in [-1]:
            for age in self.data_loader.knn_age_iterator():

                age_index = self.data_loader.get_age_index(age)                

                if np.sum(age_index) == 0:
                    pb.update(1)
                    continue
                data = self.data_loader.data.loc[age_index, self.data_loader.z_column_list]

                mdl = OutlierCurveKNN(age, gender, self.model_path)
                if not self.train_mode:
                    mdl = mdl.load()

                dens_values = mdl.neighborhood(data, fit=self.train_mode)
                self.data_loader.data.loc[age_index, "curve_density"] = dens_values

                z_scores_age = mdl.z_scores(dens_values, fit=self.train_mode)
                self.data_loader.data.loc[age_index, "curve_z_age"] = z_scores_age

                if self.train_mode:
                    mdl.distribution_plot(dens_values, z_scores_age)
                    mdl.save()

                pb.update(1)

        mdl_total = OutlierCurveKNN(0, -1, self.model_path)
        if not self.train_mode:
            mdl_total = mdl_total.load()

        gender_density = self.data_loader.data["curve_density"].values
        gender_z_scores = mdl_total.z_scores(gender_density, fit=self.train_mode)

        if self.train_mode:
            mdl_total.distribution_plot(gender_density, gender_z_scores)
            mdl_total.save()
        self.data_loader.data["curve_z_total"] = gender_z_scores
   

    def calculate_shift(self):

        self.data_loader.add_zero_columns([self.best_age_fit_column, self.shift_column, self.age_fit_z_score,
                                           self.age_specific_shift_column, self.abs_shift_column])

        num_items = self.data_loader.data.shape[0]

        progress_bar = tqdm.tqdm(desc=f"Calculating shifts for {num_items} items.", total=num_items)

        shift_differences = []

        for idx, item in self.data_loader.data.iterrows():
            subject_values = [item[f'{tmp_a}_{self.z_score_column}'] for tmp_a in self.data_loader.knn_range]
            max_assimilation_idx = np.argmax(subject_values)
            # todo: is that the best thing to do?
            diff_to_best_age_fit = np.abs(subject_values[max_assimilation_idx] - item[self.z_score_column])
            shift_differences.append(diff_to_best_age_fit)
            if diff_to_best_age_fit > 0.05:  
                best_age_fit = self.data_loader.knn_range[max_assimilation_idx]
                shift = best_age_fit - item[self.data_loader.specifier.age_encoded]
                best_age_fit_z_score = subject_values[max_assimilation_idx]                
            else:
                best_age_fit = item[self.data_loader.specifier.age_encoded]
                shift = 1e-3
                best_age_fit_z_score = item[self.z_score_column]
                # print(f"Skipping age change with {diff_to_best_age_fit}")
            self.data_loader.data.at[idx, self.age_fit_z_score] = best_age_fit_z_score
            self.data_loader.data.at[idx, self.best_age_fit_column] = best_age_fit
            self.data_loader.data.at[idx, self.shift_column] = shift
            self.data_loader.data.at[idx, self.abs_shift_column] = np.abs(shift)
            progress_bar.update(1)
        progress_bar.close()


    def plot_age_range(self):
        if self.plot:            
            n_rows = 4
            fig, ax_tuple = plt.subplots(figsize=(17, 13), nrows=n_rows, ncols=1)

            age_data = [self.data_loader.get_age_data(a, self.best_age_fit_column)
                        for a in self.data_loader.knn_range]
            ax_tuple[0].boxplot(age_data)
            ax_tuple[0].set_xticklabels(self.data_loader.knn_range)
            ax_tuple[0].set_xlabel(self.best_age_fit_column)

            z_score_data = [self.data_loader.get_age_data(a, self.shift_column)
                            for a in self.data_loader.knn_range]
            ax_tuple[1].boxplot(z_score_data)
            ax_tuple[1].set_xticklabels(self.data_loader.knn_range)
            ax_tuple[1].set_xlabel(self.shift_column)

            shift_age_data = [self.data_loader.get_age_data(a, "curve_z_total")
                              for a in self.data_loader.knn_range]
            ax_tuple[2].boxplot(shift_age_data)
            ax_tuple[2].set_xlabel("curve_z_total")
            ax_tuple[2].set_xticklabels(self.data_loader.knn_range)

            bag_data = [self.data_loader.get_age_data(a, "z_score")
                        for a in self.data_loader.knn_range]
            ax_tuple[3].boxplot(bag_data)
            ax_tuple[3].set_xlabel("z_score")
            ax_tuple[3].set_xticklabels(self.data_loader.knn_range)

            # ax3.plot(np.arange(0, len(age_range), 1), count_values)
            plt.savefig(self.get_plot_path().joinpath('shift_over_age_range.png'))
            plt.close()

  
    def inspect(self):

        self.data_loader.correlate_controls()

        if self.plot:
            self.get_plot_path().joinpath('age_in_other_age_models').mkdir(exist_ok=True)

        for alter in self.data_loader.knn_age_iterator():
            age_data = self.data_loader.get_age_data(alter)

            if age_data.shape[0] == 0:
                continue

            comparative_values = [age_data[f'{tmp_a}_{self.z_score_column}'] for tmp_a in self.data_loader.knn_range]            

            if self.plot:

                fig, ax = plt.subplots()
                ax.boxplot(comparative_values)
                ax.set_xticklabels(self.data_loader.knn_range)
                ax.axvline(x=list(self.data_loader.knn_range).index(alter), color='r')
                plt.title(f"{alter} | n={age_data.shape[0]}")
                plt.savefig(self.get_plot_path().joinpath(f'age_in_other_age_models/{alter}.png'))
                plt.close()
