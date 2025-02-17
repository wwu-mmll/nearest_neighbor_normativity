import pprint
import pandas as pd
import numpy as np


from data import *
from shared import RESULTS_FOLDER, DataLoader
from real_anova import AnovaParalellizer, AnovaCase



def run_analysis(dataloaders, target_variables,  n_perms: int = 0, random_seed: int = 234456,
                 n_processes: int = 1):

    parallel = n_processes > 1
    permutation = n_perms > 0
    anova_cases = list()
    for dl in dataloaders:
        tmp_dl = dl() if not isinstance(dl, DataLoader) else dl
        for diagnosis_name, diagnosis_code in tmp_dl.specifier.disease_dict.items():
            data_loader = dl() if not isinstance(dl, DataLoader) else dl
            data_loader.load_data(data_spec='knn_data')
            data_loader.data = data_loader.get_data_for_controls_and_(diagnosis_name, match=False)
            data_loader.data[data_loader.specifier.age_encoded + "_squared"] = np.square(data_loader.data[data_loader.specifier.age_encoded])
            data_loader.data.loc[:, data_loader.specifier.diagnosis_encoded] = data_loader.data[data_loader.specifier.diagnosis_encoded].map({'HC': 0, diagnosis_name: 1})
            data_loader.data.loc[:, data_loader.specifier.sex_encoded] -= 1

            if permutation:
                # generate permutations
                true_variable = data_loader.data[data_loader.specifier.diagnosis_encoded]
                np.random.seed(random_seed)
                permutations = np.asarray([np.random.permutation(true_variable) for _ in range(n_perms)])
                permutations = permutations.reshape((true_variable.shape[0], n_perms))
    

            for target_variable in target_variables:
                anova_columns = [target_variable, 
                                 data_loader.specifier.age_encoded, 
                                 data_loader.specifier.sex_encoded, 
                                 data_loader.specifier.diagnosis_encoded]
                anova_data = data_loader.data[anova_columns].copy()

                infos = {'target': target_variable,
                         'diagnosis': diagnosis_name,
                         'n_samples': anova_data.shape[0],
                         'bootstrap': -1}
                predictors = {i: False if i in [data_loader.specifier.age_encoded] else True for i
                              in anova_columns if i not in [target_variable]}

                odp = RESULTS_FOLDER / 'correlation' / diagnosis_name
                ac = AnovaCase(anova_data,
                               endog_var=target_variable,
                               predictors=predictors,
                               infos=infos,
                               exog_var_of_interest=data_loader.specifier.diagnosis_encoded,
                               case_output_dir= odp / f'GT__{diagnosis_name}__{target_variable}.csv')
                if parallel:
                    anova_cases.append(ac)
                else:
                    output = ac.run()
                    print(output)

                if permutation:
                    for pidx in range(n_perms):
                        infos_perm = dict(infos)
                        infos_perm["bootstrap"] = pidx
                        anova_data_perm = anova_data.copy()
                        anova_data_perm[data_loader.specifier.diagnosis_encoded] = permutations[:, pidx]
                        ac_perm = AnovaCase(anova_data_perm,
                                            endog_var=target_variable,
                                            predictors=predictors,
                                            infos=infos_perm,
                                            exog_var_of_interest=data_loader.specifier.diagnosis_encoded,
                                            case_output_dir= odp / f'perm{str(pidx).zfill(4)}__{diagnosis_name}__{target_variable}.csv')
                        if parallel:
                            anova_cases.append(ac_perm)
                        else:
                            output_perm = ac_perm.run()
                            print(output_perm)

    ap = AnovaParalellizer(output_folder=RESULTS_FOLDER / 'correlation')
    if parallel:
        ap.run_anovas_in_parallel(anova_cases)
    ap.collect_outputs()
    if permutation:
        ap.evaluate_perm_p_levels()


def collect_results(dataloaders, target_variables):
    collected_result_df = None
    for dl in dataloaders:
        data_loader = dl() if not isinstance(dl, DataLoader) else dl
        for name, diagnosis_code in data_loader.specifier.disease_dict.items():
            for target_variable in target_variables:
                result_df = pd.read_csv(
                    RESULTS_FOLDER / 'correlation' / name / data_loader.specifier.name / target_variable / f'statistics_{data_loader.specifier.name}_{target_variable}.csv')
                result_df["study"] = [data_loader.specifier.name] * result_df.shape[0]
                result_df["disease"] = [name] * result_df.shape[0]
                collected_result_df = result_df if collected_result_df is None else pd.concat(
                    [collected_result_df, result_df], axis=0)
    collected_result_df.sort_values(by=["predictor", "disease", "target"], inplace=True)
    return collected_result_df


if __name__ == '__main__':

    run_analysis([ADMCIDataLoader, NIFDDataLoader],
                 ['curve_z_total', 'BAG_SVM', 'Z_num', 'Z_sum'],
                 n_perms=1000, n_processes=4)

















