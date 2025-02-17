from datetime import datetime
import pandas as pd
import pcntoolkit as pcn
from pathlib import Path
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from data import *
from shared import DataLoader, ROOT_FOLDER, BRAIN_MEASURES


class NormativePreparation:

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.normative_folder = self.data_loader.specifier.data_folder / 'normative'
        if not self.normative_folder.exists():
            self.normative_folder.mkdir(parents=True, exist_ok=True)

        self.name = self.data_loader.specifier.name.lower()
        self.feature_names = ["Z_GM", "Z_WM", "Z_CSF", "Z_WMH"]
        self.data_loader.load_data(data_spec='knn_data')

        self.covariates_train_file: Path = self.normative_folder / f'{self.name}_covariates.txt'
        self.features_train_file: Path = self.normative_folder / f'{self.name}_features.txt'
        self.covariates_test_file: Path = self.normative_folder / f'{self.name}_covariates_test.txt'
        self.features_test_file: Path = self.normative_folder / f'{self.name}_features_test.txt'

    def prepare_features(self, overwrite_files: bool = False, create_testset: bool = False):

        if overwrite_files or (not self.covariates_train_file.exists() or not self.features_train_file.exists()):
            covariates = self.data_loader.data[[self.data_loader.specifier.age_encoded,
                                                self.data_loader.specifier.sex_encoded]]
            features = self.data_loader.get_brain_measures()
            # correct for TIV
            tiv = features["TIV"]
            features.drop(columns=["TIV"], inplace=True)
            # print(features.columns)
            for c in features.columns:
                features[c] = features[c] / tiv

            if create_testset:
                train_idx, test_idx = next(iter(ShuffleSplit(n_splits=1,
                                                             test_size=0.2,
                                                             random_state=1337).split(covariates)))

                # StandardScaler().fit_transform()
                self.write_to_text_file(covariates.iloc[train_idx],
                                        self.covariates_train_file)
                self.write_to_text_file(covariates.iloc[test_idx],
                                        self.covariates_test_file)
                self.write_to_text_file(features.iloc[train_idx],
                                        self.features_train_file)
                self.write_to_text_file(features.iloc[test_idx],
                                        self.features_test_file)
            else:
                self.write_to_text_file(covariates, self.covariates_train_file)
                self.write_to_text_file(features, self.features_train_file)

    @staticmethod
    def write_to_text_file(dataframe, path_to_save):
        dataframe.to_csv(path_to_save, sep=' ', header=False, index=False)


class NormativeModel:

    MODEL_FOLDER = "./Models"

    def __init__(self):
        pass

    def train(self):
        data_loader = DataLoader.load_knn_train_data()
        data_preparations = NormativePreparation(data_loader)
        data_preparations.prepare_features(overwrite_files=True, create_testset=True)

        print("Starting Estimation:", datetime.now())
        pcn.normative.estimate(covfile=data_preparations.covariates_train_file.as_posix(),
                               respfile=data_preparations.features_train_file.as_posix(),
                               testcov=data_preparations.covariates_test_file.as_posix(),
                               testresp=data_preparations.features_test_file.as_posix(),
                               # alg='gpr',
                               alg='blr',
                               optimizer='powell',
                               inscaler='standardize',
                               outscaler='standardize',
                               outputsuffix='_train',
                               savemodel=True)

    def predict(self, data_loader):
        predict_preparations = NormativePreparation(data_loader)
        predict_preparations.prepare_features(overwrite_files=True)

        yhat_te, s2_te, Z = pcn.normative.predict(covfile=predict_preparations.covariates_train_file.as_posix(),
                                                  respfile=predict_preparations.features_train_file.as_posix(),
                                                   # alg='blr',
                                                   # optimizer='powell',
                                                   alg='blr',
                                                   inputsuffix='_train',
                                                   inscaler='standardize',
                                                   outscaler='standardize',
                                                   model_path=self.MODEL_FOLDER)

        data_loader.data[predict_preparations.feature_names] = Z
        data_loader.data["Z_num"] = (data_loader.data[predict_preparations.feature_names].abs() > 1.96).sum(axis=1)
        data_loader.data["Z_sum"] = (data_loader.data[predict_preparations.feature_names].abs()).sum(axis=1)
        data_loader.overwrite_csv("knn_data")


if __name__ == '__main__':

    trainer = NormativeModel()
    trainer.train()

    for dl in [
               FOR2107DataLoader,
               ADNIDataLoader, 
               AIBLDataLoader,
               NIFDDataLoader, 
               OASISDataLoader               
               ]:  
        print(dl().specifier.name)
        trainer.predict(dl())
