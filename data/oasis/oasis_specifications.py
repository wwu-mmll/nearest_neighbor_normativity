import numpy as np
import pandas as pd
from brainage.shared import DATA_FOLDER, DataLoader, Specifications

OASIS_DATA = DATA_FOLDER.joinpath('oasis3/oasis3_joined.csv')
OASIS_KNN_RESULTS_DATA = DATA_FOLDER.joinpath('oasis3/oasis_knn.csv')

OASIS_SPECIFICATIONS = Specifications('OASIS3')

with OASIS_SPECIFICATIONS as a:
    a.age_column = 'age at visit'
    a.age_needs_rounding = True

    a.female_identifier = 2
    a.sex_column = 'GENDER'

    a.diagnosis_column = 'diagnosis'
    a.healthy_identifier = 0
    a.disease_dict = {"MCI": 0.5, "AD": 1}

    a.site = "site"

    a.raw_data = OASIS_DATA
    a.knn_data = OASIS_KNN_RESULTS_DATA

    a.id_column = 'OASISID'
    a.filter_duplicates_by = a.id_column


class OASISDataLoader(DataLoader):

    def __init__(self):
        super().__init__(specifier=OASIS_SPECIFICATIONS)


