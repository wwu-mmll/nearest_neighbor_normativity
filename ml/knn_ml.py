import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pymodm.errors
import seaborn as sns
from sklearn.decomposition import PCA
from photonai import PermutationTest
from photonai.processing.results_handler import MDBHyperpipe
from pymodm import connect
from shared import PLOT_PATH, ROOT_FOLDER, RESULTS_FOLDER, DataLoader, Specifications
from data import *

import warnings
warnings.filterwarnings("ignore")

N_PERMS = 1000

class AnalysisCase:

    def __init__(self, analysis_name: str = '',  predictor_name: str = '', feature_list=None):
        self.analysis_name = analysis_name
        self.predictor_name = predictor_name
        self.feature_list = feature_list if feature_list is not None else [self.predictor_name]
        self._disease_name = ''
        self._perm_id = None

    def get_features(self, data):
        if len(self.feature_list) == 1:
            features = data[self.feature_list[0]].values.reshape((-1, 1))
        else:
            features = data[self.feature_list].values
        return features

    @property
    def disease_name(self):
        return self._disease_name

    @disease_name.setter
    def disease_name(self, value):
        self._disease_name = value
        self._perm_id = f'{self.analysis_name}-{self._disease_name}-{self.predictor_name}'

    @property
    def perm_id(self):
        return self._perm_id


class MLAnalysis:

    def __init__(self, analysis_name, data_loader_cls, plot=False, ml=True, perm=False, restart_perm=False):
        self.analysis_name = analysis_name
        self.data_loader = data_loader_cls() if not isinstance(data_loader_cls, DataLoader) else data_loader_cls
        self.plot = plot
        self.ml = ml
        self.perm = perm
        self.restart_perm = restart_perm

        self.plot_path = None
        self.result_list = []
        self.project_folder = ROOT_FOLDER.joinpath('ml_analysis/classification')
        self.project_folder.mkdir(parents=True, exist_ok=True)
        self.mongodb_path = 'mongodb://localhost:27017/photon_results'

    def get_hyperpipe(self):
        from sklearn.model_selection import StratifiedKFold
        from photonai import ClassificationPipe, PipelineElement, OutputSettings
        from photonai.optimization import MinimumPerformanceConstraint

        if self.perm:
            add_dict = {'output_settings': OutputSettings(mongodb_connect_url='mongodb://localhost:27017/photon_results')}
        else:
            add_dict = {'project_folder': self.project_folder}

        clf_pipe = ClassificationPipe(self.analysis_name,
                                      outer_cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=1337),
                                      inner_cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=1337),
                                      add_default_pipeline_elements=False,
                                      verbosity=-1,
                                      best_config_metric="balanced_accuracy",
                                      performance_constraints=[MinimumPerformanceConstraint('f1_score',
                                                                                             0.35, 'mean'),
                                                               MinimumPerformanceConstraint('balanced_accuracy',
                                                                                            0.4, 'mean')
                                                               ],
                                      **add_dict
                                      )

        clf_pipe += PipelineElement("ImbalancedDataTransformer",
                                    hyperparameters={'method_name': ['RandomUnderSampler',
                                                                     "SMOTE",
                                                                     ]},                                    
                                    )        
        clf_pipe += PipelineElement('StandardScaler')
        clf_pipe += PipelineElement("SVC",
                                    hyperparameters={'C': [0.5, 1, 5, 10, 100, 500, 1000, 5000],
                                                     'gamma': ['auto', 'scale']
                                    },                                    
                                    max_iter=int(1e06))
        return clf_pipe

    def prepare(self):

        self.data_loader.load_data(data_spec="knn_data", apply_filter=True)                
        self.plot_path = PLOT_PATH.joinpath(f'ml_plots/{self.analysis_name}')
        self.plot_path.mkdir(parents=True, exist_ok=True)

    def run(self):
        cases = {
            'curve_z_total': AnalysisCase(self.analysis_name, 'curve_z_total'),
            'BAG_SVM': AnalysisCase(self.analysis_name, 'BAG_SVM'),
            'Z_sum': AnalysisCase(self.analysis_name, predictor_name="Z_sum"),
            'Z_num': AnalysisCase(self.analysis_name, predictor_name="Z_num"),
        }

        self.prepare()        
        output_records = []
        if self.perm:
            connect(self.mongodb_path, alias="photon_core")

        for disease_name, disease in self.data_loader.specifier.disease_dict.items():
            group_data = self.data_loader.get_data_for_controls_and_(disease_name, match=False)            

            target = np.zeros((group_data.shape[0]))
            target[group_data[self.data_loader.specifier.diagnosis_encoded] != "HC"] = 1

            for analysis_id, analysis_item in cases.items():

                print(f'{"*" * 35} {disease_name} <- {analysis_item.predictor_name} {"*" * 35}')
                analysis_item.disease_name = disease_name
                features = analysis_item.get_features(group_data)
                not_nan_index = np.squeeze(~np.isnan(features))
                loc_features = np.copy(features[not_nan_index])
                loc_target = np.copy(target[not_nan_index])

                if self.ml:
                    self.fit_ml_model(loc_features, loc_target, analysis_item)
                    self.save_results()
                
  
    def fit_ml_model(self, features, targets, analysis_case):
        print("Fit ML Model...")
        clf_pipe = self.get_hyperpipe()
        clf_pipe.fit(features, targets)

        print(clf_pipe.results.best_config.human_readable_config)
        results = {"disease": analysis_case.disease_name, "predictor_set": analysis_case.predictor_name}
        # add it in a specific order so that it is pretty in csv
        for m in ["recall", "specificity", "precision", "f1_score", "balanced_accuracy"]:
            metric_val = clf_pipe.results.get_test_metric(m)
            metric_std = clf_pipe.results.get_test_metric(m, "std")
            results[m] = metric_val
            results[f"{m}_std"] = metric_std
            print(m, metric_val, metric_std)

        self.result_list.append(results)

    def save_results(self):
        result_df = pd.DataFrame.from_records(self.result_list)
        result_df.sort_values(by=["disease", "balanced_accuracy"], inplace=True)
        result_df.to_csv(ROOT_FOLDER.joinpath(f"results/knn_ml_results_{self.analysis_name}.csv"))


def join_csvs(use_permutations=False):
    result_csvs = RESULTS_FOLDER.glob('*.csv')
    results_df = None
    for result_csv in result_csvs:
        if not use_permutations and result_csv.name.startswith('permutation'):
            continue
        name_of_study = result_csv.stem.split('_')[-1]
        loaded_df = pd.read_csv(result_csv)
        loaded_df["study"] = [name_of_study] * loaded_df.shape[0]
        results_df = loaded_df if results_df is None else pd.concat([results_df, loaded_df],
                                                                     ignore_index=True, axis=0)

    results_df.to_csv(RESULTS_FOLDER.parent.joinpath('ml_results_joined_nlb.csv'), index=False)


if __name__ == '__main__':

    neuro_dict = {
        'admci': ADMCIDataLoader,
        'nifd': NIFDDataLoader,              
                  }

    for name, loader in neuro_dict.items():
       loaded_loader = loader()
       analyser = MLAnalysis(name, loader, plot=False, ml=True, perm=False, restart_perm=False)
       analyser.run()
    join_csvs()    

