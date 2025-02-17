import pandas as pd
import numpy as np
import pprint
import matplotlib.pyplot as plt
from data import *
from shared import DataLoader, PLOT_PATH
from model.outlier_knn import OutlierKNN
from sklearn.model_selection import KFold
from pingouin import intraclass_corr


if __name__ == '__main__':


    dl_train = DataLoader.load_knn_train_data()
    dl_train.load_data()

    dl_adni = ADNIDataLoader()
    dl_adni.load_data(controls_only=True)

    dl_for2107 = FOR2107DataLoader()
    dl_for2107.load_data(controls_only=True)

    dl_joined = DataLoader.join_and_create(dl_adni, dl_for2107, "test_stability")

    ages_to_inspect = dl_joined.knn_range
    divider_list = [10, 5, 3, 2]
    max_nun_increments = len(divider_list)
    min_num_train_samples = 250
    min_num_test_samples = 10

    icc_dict = {}
    record_list = []
    for gender in dl_train.gender_iterator():
        for age in ages_to_inspect:

            icc_dict[age] = {}

            train_index = dl_train.get_age_gender_index_for_train(age, gender)
            train_data = dl_train.get_brain_measures(train_index)
            test_index = dl_joined.get_age_gender_index(age, gender)
            test_data = dl_joined.get_brain_measures(test_index)

            num_train_samples = np.sum(train_index)
            num_test_samples = np.sum(test_index)

            print(f"age: {age}, train samples: {num_train_samples}, test samples: {num_test_samples}")

            if num_train_samples < min_num_train_samples or num_test_samples < min_num_test_samples:
                continue

            age_data = train_data[train_index]

            for ssi, num_folds in enumerate(divider_list):
                num_train_samples_fold = np.floor(num_train_samples / num_folds)
                print(f"age: {age}, num test subjects: {num_test_samples}, "
                      f"num train samples: {num_train_samples_fold}")

                k_fold_item = KFold(n_splits=num_folds, shuffle=True, random_state=2909)
                z_scores = np.zeros((num_test_samples, num_folds))
                for fidx, (_, rep_index) in enumerate(k_fold_item.split(age_data)):
                    sample_size_train_data = train_data.iloc[rep_index]
                    knn_model = OutlierKNN(age, 0, save_models_to='./tmp')
                    output = knn_model.norm_neighborhood(sample_size_train_data, False)
                    test_distances = knn_model.neighborhood(test_data, False)
                    test_z_scores = knn_model.z_scores(test_distances)
                    z_scores[:, fidx] = test_z_scores

                fcking_columns = ["measurement", "rater", "z_score"]
                new_fcking_df = pd.DataFrame(columns=fcking_columns)
                
                print("copy")
                for subject in range(z_scores.shape[0]):
                    subject_data = z_scores[subject]
             
                    for measurement_idx in range(z_scores.shape[1]):
                        value = subject_data[measurement_idx]
                        new_fcking_df.loc[-1] = [measurement_idx, subject, value]
                        new_fcking_df.index = new_fcking_df.index + 1

                print("icc")
                icc_pingouin = intraclass_corr(data=new_fcking_df,
                                               targets="rater",
                                               raters="measurement",
                                               ratings="z_score")
                icc_pingouin.set_index("Type", inplace=True)
                icc_val = icc_pingouin.loc['ICC2', 'ICC']
                print(icc_val)
                icc_dict[age][num_folds] = (icc_val, num_train_samples_fold)

            pp = pprint.PrettyPrinter(indent=4)  
            pp.pprint(icc_dict)

        y_values = [[age_dict.get(number)[0] for age_dict in icc_dict.values() if len(age_dict) > 0]
                    for number in divider_list]
        x_values = [[age_dict.get(number)[1] for age_dict in icc_dict.values() if len(age_dict) > 0]
                    for number in divider_list]

        fig, ax = plt.subplots()
        for idx, ssitem_plot in enumerate(divider_list):
            ax.scatter(x_values[idx], y_values[idx], label=ssitem_plot)

            for x, y in zip(x_values[idx], y_values[idx]):
                record_list.append({'sample_size': x,
                                    'icc': y,
                                    'gender': gender,
                                    'division': ssitem_plot})

        ax.set_xlabel("Sample Size")
        ax.set_ylabel(f" ICC(2,1)")
        ax.set_ylim(0, 1.25)
        plt.legend()
        plt.title(f"Stability analysis")
        plt.savefig(dl_train.specifier.plot_folder.joinpath(f'icc2_{gender}.png'))

    df = pd.DataFrame.from_records(record_list)
    df.to_csv(PLOT_PATH / 'paper' / 'icc2-1.csv')






