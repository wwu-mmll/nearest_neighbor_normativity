import pandas as pd
import numpy as np
from data import *
from shared import DATA_FOLDER, DataLoader, Specifications


ADMCI_KNN_RESULTS_DATA = DATA_FOLDER.joinpath('admci/admci_knn.csv')

ADMCI_SPECIFICATIONS = Specifications('ADMCI')
with ADMCI_SPECIFICATIONS as a:    
    a.disease_dict = {"MCI": "MCI", "AD": "AD"}

    a.raw_data = ADMCI_KNN_RESULTS_DATA
    a.knn_data = ADMCI_KNN_RESULTS_DATA

    a.id_column = 'jid'
    a.site = 'jsite'
    a.norm_variables = False


class ADMCIDataLoader(DataLoader):

    def __init__(self):
        super().__init__(specifier=ADMCI_SPECIFICATIONS)


if __name__ == "__main__":
    neuro_dict = {
        'oasis3': OASISDataLoader,
        'adni': ADNIDataLoader,
        'aibl': AIBLDataLoader,         
        }
     
    joined_df = None
    for _, data_loader in neuro_dict.items():
        loader = data_loader()
        print(loader.specifier.name)
        loader.load_data('knn_data')
        loader.data["jid"] = [f'{loader.specifier.name}_{id}' for id in loader.data[loader.specifier.id_column]]
        loader.data["jsite_pre"] = [f'{loader.specifier.name}_{site}' for site in loader.data[loader.specifier.site]]
        loader.data["z_score_squared"] = np.sign(loader.data["z_score"]) * np.square(loader.data["z_score"])
        joined_df = loader.data if joined_df is None else pd.concat([joined_df, loader.data],
                                                                    axis=0,
                                                                    ignore_index=True)
    joined_df["jsite"] = joined_df["jsite_pre"].astype("category").cat.codes
    joined_df.to_csv(ADMCI_KNN_RESULTS_DATA, index=False)

    
    