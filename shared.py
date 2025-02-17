import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

DATA_FOLDER = Path('/space/neighbors/')
ROOT_FOLDER = Path('/home/rleenings/Projects/Neighbors/')
PLOT_PATH = ROOT_FOLDER.joinpath('plots')

KNN_TRAIN_DATA = DATA_FOLDER.joinpath('merged_studies.csv')
NAKO2022_DATA = DATA_FOLDER.joinpath('nako/nako2022_brain_measures.csv')
RESULTS_FOLDER = ROOT_FOLDER.joinpath('results')
KNN_MODEL_FOLDER = ROOT_FOLDER.joinpath('knn_models')

BRAIN_MEASURES = ["TIV", "GM", "WM", "CSF", "WMH"]


class Specifications:

    def __init__(self, name):
        self.name = name

        self.age_column = 'age'
        self.sex_column = 'sex'
        self.female_identifier = 2
        self.site = ''

        self.diagnosis_column = ''
        self.healthy_identifier = ''
        self.disease_dict = {}

        self.raw_data = ''
        self.knn_data = ''

        self.id_column = ''
        self.filter_duplicates_by = None
        self.norm_variables = True

    @property
    def diagnosis_encoded(self):
        return "diagnosis_encoded"

    @property
    def sex_encoded(self):
        return "sex_encoded"

    @property
    def age_encoded(self):
        return "age_encoded"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            pass
        else:
            print('An error occurred.')
        return True

    @property
    def plot_folder(self):
        return PLOT_PATH.joinpath(f'{self.name}/')

    @property
    def data_folder(self):
        return DATA_FOLDER / self.name.lower()


class NAKO22Spec(Specifications):
    def __init__(self, name):
        super(NAKO22Spec, self).__init__(name)
        self.age_column = 'basis_age'
        self.sex_column = 'basis_sex'
        self.female_identifier = 2
        self.site = "StudZ"

        self.raw_data = NAKO2022_DATA
        self.knn_data = DATA_FOLDER.joinpath("nako22_knn.csv")


class KNNTrainData(Specifications):

    def __init__(self, name):
        super(KNNTrainData, self).__init__(name)
        self.age_column = 'age'
        self.sex_column = 'sex'
        self.female_identifier = 2

        self.raw_data = KNN_TRAIN_DATA
        self.knn_data = DATA_FOLDER.joinpath("merged_studies_knn.csv")


class DataLoader:
    lowest_valid_age = 21
    highest_valid_age = 72
    knn_range = np.arange(lowest_valid_age, highest_valid_age + 1)
    z_column_list = [f'{tmp_a}_z_score' for tmp_a in knn_range]    

    def __init__(self, specifier):
        self.specifier = specifier
        self.data = None
        self.age_range = None
        self.genders = None

        if self.specifier.name:
            self.specifier.plot_folder.mkdir(parents=True, exist_ok=True)

    def load_data(self, data_spec: str = 'raw_data',
                  apply_filter: bool = False,
                  filter_duplicates: bool = False,
                  knn_age_range: bool = True,
                  controls_only: bool = False,
                  conform_demos: bool = False):

        self.data = pd.read_csv(getattr(self.specifier, data_spec))

        if self.specifier.norm_variables:
            self.setup_demos()

        if apply_filter:
            if self.specifier.filter_duplicates_by is not None:
                self.data = self.data[self.data.groupby(self.specifier.filter_duplicates_by).cumcount() == 0]
            self.data = self.apply_filter(self.data)

        if knn_age_range:

            self.data = self.data[(self.data[self.specifier.age_encoded] >= DataLoader.lowest_valid_age) &
                                  (self.data[self.specifier.age_encoded] <= DataLoader.highest_valid_age)]            

        if controls_only:
            self.data = self.get_controls()

    def setup_demos(self):

        self.data.dropna(subset=[self.specifier.age_column, self.specifier.sex_column],
                         how="any",
                         inplace=True)
        if self.specifier.diagnosis_column:
            self.data.dropna(subset=[self.specifier.diagnosis_column], inplace=True)

        self.data["age_encoded"] = np.round(self.data[self.specifier.age_column]).astype(int)
        self.age_range = self.data[self.specifier.age_encoded].unique().astype(int)
        self.age_range.sort()

        self.encode_gender()
        self.genders = self.data[self.specifier.sex_encoded].unique()

        self.encode_diagnosis()

    def encode_gender(self):
        self.data[self.specifier.sex_encoded] = [2 if i == self.specifier.female_identifier else 1
                                                 for i in self.data[self.specifier.sex_column]]

    def encode_diagnosis(self):
        if len(self.specifier.disease_dict) > 0:
            inv_map = {v: k for k, v in self.specifier.disease_dict.items()}
            self.data[self.specifier.diagnosis_encoded] = ["HC" if i == self.specifier.healthy_identifier else
                                                           inv_map[i] for i in self.data[self.specifier.diagnosis_column]]
        else:
            self.data[self.specifier.diagnosis_encoded] = ["HC"] * self.data.shape[0]

    def age_iterator(self):
        for a in self.age_range:
            yield a

    def knn_age_iterator(self):
        for knn_a in self.knn_range:
            yield knn_a

    def gender_iterator(self):
        for g in self.genders:
            yield g

    def get_age_index(self, age):
            return self.data[self.specifier.age_encoded] == age

    def get_gender_index(self, gender):
        return self.data[self.specifier.sex_encoded] == gender

    def get_age_gender_index(self, age, gender):
        return (self.get_age_index(age)) & (self.get_gender_index(gender))

    def get_age_gender_index_for_train(self, current_age, gender, max_expected_sample_size=500):

        age_bin_below, age_bin_above = 0, 0

        min_age = np.min(self.age_range)
        max_age = np.max(self.age_range)
        actual_number_of_age_samples = np.sum(self.get_age_gender_index(current_age, gender))
        number_of_samples_below = 0
        number_of_samples_above = 0
        if current_age - 1 > min_age:
            number_of_samples_below = np.sum(self.get_age_gender_index(current_age - 1, gender))
        if current_age + 1 < max_age:
            number_of_samples_above = np.sum(self.get_age_gender_index(current_age + 1, gender))

        current_sample_size = actual_number_of_age_samples

        if number_of_samples_below > 0:
            below_distance_with = np.abs(max_expected_sample_size - (actual_number_of_age_samples + number_of_samples_below))
            below_distance_without = np.abs(max_expected_sample_size - actual_number_of_age_samples)
            if below_distance_with < below_distance_without:
                age_bin_below = 1
                current_sample_size += number_of_samples_below

        if number_of_samples_above > 0:
            above_distance_with = np.abs(max_expected_sample_size - (current_sample_size + number_of_samples_above))
            above_distance_without = np.abs(max_expected_sample_size - current_sample_size)
            if above_distance_with < above_distance_without:
                age_bin_above = 1
                current_sample_size += number_of_samples_above

        lower_age_bound = max(min_age, current_age - age_bin_below)
        upper_age_bound = min(max_age, current_age + age_bin_above)
        age_gender_index = ((self.data[self.specifier.age_encoded] >= lower_age_bound) &
                            (self.data[self.specifier.age_encoded] <= upper_age_bound) &
                            (self.data[self.specifier.sex_encoded] == gender))

        num_samples_age_gender_train_index = np.sum(age_gender_index)
        print(current_age, actual_number_of_age_samples, number_of_samples_below, number_of_samples_above,
              age_bin_below, age_bin_above, num_samples_age_gender_train_index)

        return age_gender_index

    def get_age_data(self, age, column=None):
        if column is None:
            return self.data[self.get_age_index(age)]
        else:
            return self.data.loc[self.get_age_index(age), column]

    def get_gender_data(self, gender):
        return self.data[self.get_gender_index(gender)]

    def get_age_gender_data(self, age, gender, column=None):
        if column is None:
            return self.data[self.get_age_gender_index(age, gender)]
        else:
            return self.data.loc[self.get_age_gender_index(age, gender), column]

    def add_zero_columns(self, columns_to_add):
        fake_news = np.full((self.data.shape[0], len(columns_to_add)), np.nan)
        new_placeholder_df = pd.DataFrame(data=fake_news, columns=columns_to_add)
        self.data = pd.concat((self.data, new_placeholder_df), axis=1)

    def get_brain_measures(self, specific_index=None):        
        if specific_index is None:
            data = self.data.loc[:,BRAIN_MEASURES]
        else:
            data = self.data.loc[specific_index, BRAIN_MEASURES]

        data.rename(columns={self.specifier.sex_encoded: 'sex'}, inplace=True)
        return data

    def get_z_scores(self, specific_index=None):
        if specific_index is None:
            data = self.data.loc[:, self.z_column_list]
        else:
            data = self.data.loc[specific_index, self.z_column_list]
        return data

    def overwrite_csv(self, data_spec: str = 'knn_data'):
        file = getattr(self.specifier, data_spec)
        if file:
            self.data.to_csv(file, index=False)
        else:
            print(f"No {data_spec} file specified for {self.specifier.name}. Skipping overwrite")

    def encode_sex(self):
        self.data[self.specifier.sex_encoded] = [2 if i == self.specifier.female_identifier else 1
                                                 for i in self.data[self.specifier.sex_column]]

    def get_controls(self):
        return self.data[self.data[self.specifier.diagnosis_encoded] == "HC"]

    def get_data_for_(self, other_group):
        relevant_data = self.data[self.data[self.specifier.diagnosis_encoded] == other_group].copy()
        return relevant_data

    def get_data_for_controls_and_(self, other_group, match=False):
        relevant_data = (self.data[(self.data[self.specifier.diagnosis_encoded] == "HC") |
                                   (self.data[self.specifier.diagnosis_encoded] == other_group)]).copy()        

        if match:
            non_HCS = relevant_data[self.specifier.diagnosis_encoded] != "HC"
            num_of_samples_in_other_group = np.sum(non_HCS)
            num_of_controls = np.abs(relevant_data.shape[0] - num_of_samples_in_other_group)

            diseased_identifier = other_group if num_of_samples_in_other_group < num_of_controls else "HC"
            healthy_identifier = "HC" if diseased_identifier != "HC" else other_group
            diseased_df = relevant_data[relevant_data[self.specifier.diagnosis_encoded] == diseased_identifier]
            healthy_df = relevant_data[relevant_data[self.specifier.diagnosis_encoded] == healthy_identifier]

            # Initialize a dictionary to keep track of matched samples
            matched_samples = {'Diseased': [], 'Healthy': []}

            # Match based on age and gender
            matched_controls = []
            for _, patient in diseased_df.iterrows():
                age_diff = np.abs(healthy_df[self.specifier.age_encoded] - patient[self.specifier.age_encoded])
                potential_matches = (healthy_df[healthy_df[self.specifier.sex_encoded] == patient[self.specifier.sex_encoded]]).copy()
                potential_matches['AgeDiff'] = age_diff[potential_matches.index]

                # Filter out potential matches that have already been used
                available_matches = potential_matches[~potential_matches[self.specifier.id_column].isin(matched_samples['Healthy'])]

                if not available_matches.empty:
                    best_match = available_matches.sort_values('AgeDiff').iloc[0]
                    matched_controls.append(best_match)

                    # Update the dictionary of matched samples
                    matched_samples['Diseased'].append(patient[self.specifier.id_column])
                    matched_samples['Healthy'].append(best_match[self.specifier.id_column])

            matched_controls = pd.DataFrame(matched_controls)
            matched_controls.drop(columns=["AgeDiff"], inplace=True)
            relevant_data = pd.concat((diseased_df, matched_controls), axis=0, ignore_index=True)

        # recalculate numbers after matching
        num_of_samples_in_other_group = np.sum(relevant_data[self.specifier.diagnosis_encoded] != "HC")
        num_of_controls = np.abs(relevant_data.shape[0] - num_of_samples_in_other_group)

        print(f"n={num_of_controls} controls and n={num_of_samples_in_other_group} of diagnosis {other_group}")

        return relevant_data

    def correlate_controls(self, controls=True):
        if self.specifier.diagnosis_column:
            relevant_data = self.get_controls() if controls else self.data
        else:
            print("Could not find specification for healthy controls. Using all data for correlations.")
            relevant_data = self.data
        return self.calc_obvious_correlations(relevant_data, self.specifier.age_encoded, self.specifier.sex_encoded)

    @staticmethod
    def apply_filter(data_to_filter):
        data = data_to_filter.dropna(subset=BRAIN_MEASURES, how="any")
        return data

    def white_list(self):
        return {'categoricals': self.CATEGORICALS,
                'numericals': self.NUMERICALS}

    def correlate_and_print(self, data, column1, column2):
        not_nan_index = (~data[column1].isna()) & (~data[column2].isna())
        corr_results = spearmanr(data.loc[not_nan_index, column1], data.loc[not_nan_index, column2])
        # print(f"{column1} and {column2} ")
        # print(corr_results)
        return corr_results

    def calc_obvious_correlations(self, data, age_column, gender_column):
        results = list()
        interesting_columns = ["z_score", "curve_z_total", "BAG_SVM", "Z_num", "Z_sum"]
        for c in interesting_columns:
            if c in data:
                print("*" * 30, c)
                res = self.correlate_and_print(data, column1=c, column2=age_column)
                # self.correlate_and_print(data, column1=c, column2=gender_column)
                results.append({'predictor': c,
                                'dataset': self.specifier.name,
                                'rho': res[0],
                                'p': res[1]})
        return results

    @staticmethod
    def create(data, name, specifier=None, setup_demos=False):
        if specifier is None:
            specifier = Specifications(name)

        data_loader = DataLoader(specifier=specifier)
        data_loader.data = data
        if setup_demos:
            data_loader.setup_demos()
        return data_loader

    @staticmethod
    def load_knn_train_data():
        data_loader = DataLoader(specifier=KNNTrainData('Train'))
        return data_loader

    @staticmethod
    def join_and_create(dl1, dl2, name):

        columns_to_keep = [dl1.specifier.sex_encoded, dl1.specifier.age_encoded] + BRAIN_MEASURES

        d1_filtered = dl1.data[columns_to_keep]
        d2_filtered = dl2.data[columns_to_keep]
        new_df = pd.DataFrame(data=np.concatenate((d1_filtered.values, d2_filtered.values)), columns=columns_to_keep)

        data_loader = DataLoader.create(new_df, name)
        return data_loader


class NAKODataLoader(DataLoader):

    def __init__(self):
        super().__init__(specifier=NAKO22Spec("NAKO"))
