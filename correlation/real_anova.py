import numpy as np
import pandas as pd
import seaborn as sns
import os
from concurrent import futures
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from pingouin import ancova, linear_regression
from pathlib import Path
from typing import Union


ANOVA_RESULTS_JOINED = 'anova_results_joined.csv'
ANOVA_PERM_DIFFS = 'perm_diff_p_values.csv'



class AnovaCase:

    def __init__(self, tmp_df, endog_var, predictors, infos, exog_var_of_interest, case_output_dir: Union[str, Path] = None):
        self.data = tmp_df
        self.endog_var = endog_var
        self.predictors = predictors
        self.infos = infos
        self.exog_var_of_interest = exog_var_of_interest
        self.case_output_path = case_output_dir 

        if not self.case_output_path.parent.exists():
            self.case_output_path.parent.mkdir(exist_ok=True, parents=True)

    def run(self):


        # scale
        for column, categorical in self.predictors.items():
            if column not in self.data.columns:
                breakpoint()
            if not categorical:
                self.data[column] = np.squeeze(StandardScaler().fit_transform(self.data[column].values.reshape((-1, 1))))

        ancova_results = ancova(self.data, dv=self.endog_var, between=self.exog_var_of_interest,
                                covar=[v for v in self.predictors.keys() if v != self.exog_var_of_interest])
        ancova_results.set_index("Source", inplace=True)
        for var in ['np2', 'p-unc', 'F', 'DF']:
            for exog in self.predictors:
                self.infos[f'{var}_{exog}'] = ancova_results[var][exog]

        y = self.data[self.endog_var]
        x = self.data.copy()
        x.drop(columns=[self.endog_var], inplace=True)
        x = x.astype(float)
        lr_output = linear_regression(x, y)
        lr_output.set_index("names", inplace=True)
        for exog in self.predictors:
            self.infos[f'beta_{exog}'] = lr_output["coef"][exog]

        if self.case_output_path is not None:
            output_df = pd.DataFrame.from_records([self.infos])
            output_df.to_csv(self.case_output_path, index=False)
        return self.infos


class AnovaParalellizer:

    def __init__(self, n_processes: int = 4,
                 output_folder: Union[str, Path] = './tmp/anova',
                 joined_file: Union[str, Path] = ANOVA_RESULTS_JOINED):
        self.n_processes = n_processes
        self.output_folder = output_folder if isinstance(output_folder, str) else Path(output_folder)
        self.joined_file_path = output_folder / joined_file if isinstance(joined_file, str) else joined_file

    @staticmethod
    def run_model(model):
        model.run()
        print(os.getpid(), " Finished ", model.predictor, flush=True)

    def run_anovas_in_parallel(self, list_of_anovas: list):
         with futures.ProcessPoolExecutor(max_workers=self.n_processes) as executor:
            executor.map(AnovaParalellizer.run_model, list_of_anovas)
    
    def collect_outputs(self):
        joined_df = pd.DataFrame()
        for d in self.output_folder.iterdir():
            if d.is_dir():
                for file in d.glob('*.csv'):
                    new_df_data = pd.read_csv(file)
                    joined_df = pd.concat([joined_df, new_df_data], axis=0, ignore_index=True)
        joined_df.to_csv(self.joined_file_path, index=False)

    def evaluate_perm_p_levels(self):

        df = pd.read_csv(self.joined_file_path)
        target_of_interest = "curve_z_total"
        col_of_interest = "np2_diagnosis_encoded"
        bootstraps = df["bootstrap"].unique()

        results = []
        for diagnosis in df['diagnosis'].unique():
            our_data = df[(df['diagnosis'] == diagnosis) & (df["target"] == target_of_interest)]
            our_data.set_index("bootstrap", inplace=True)
            our_real_value = our_data.loc[-1, col_of_interest]
            for target in df["target"].unique():
                if target == target_of_interest:
                    continue
                sub_data = df[(df['diagnosis'] == diagnosis) & (df['target'] == target)]
                if sub_data.empty:
                    raise Exception("Found no bootstraps for", diagnosis, target)
                sub_data.set_index("bootstrap", inplace=True)
                their_real_value = sub_data.loc[-1, col_of_interest]
                real_diff = our_real_value - their_real_value
                perm_diffs = np.asarray([our_data.loc[bi, col_of_interest] - sub_data.loc[bi, col_of_interest]
                                         for bi in bootstraps if bi > 0])
                sum_of_good_runs = np.sum(perm_diffs < real_diff).astype(int)
                p_value = 1.0 - (sum_of_good_runs + 1) / (len(perm_diffs) + 1)  # why +1 ?
                results.append({'diagnosis': diagnosis,
                                'target': target,
                                'diff': real_diff, 'p': p_value})
        perm_p_df = pd.DataFrame.from_records(results)
        print(perm_p_df)
        perm_p_df.to_csv(self.output_folder / ANOVA_PERM_DIFFS)










        

    
