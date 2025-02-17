from shared import DATA_FOLDER, DataLoader, Specifications

NIFD_DATA_FOLDER = DATA_FOLDER / "nifd"
NIFD_DATA = NIFD_DATA_FOLDER / "nifd_joined.csv"
NIFD_DATA_KNN = NIFD_DATA_FOLDER / "nifd_knn.csv"

NIFD_SPECIFICATIONS = Specifications('NIFD')
with NIFD_SPECIFICATIONS as a:
    a.age_column = 'Age'
    a.age_needs_rounding = True

    a.female_identifier = 2
    a.sex_column = 'GENDER'

    a.healthy_identifier = 'CON'
    a.diagnosis_column = 'DX_joined'
    a.disease_dict = {"FTD": "FTD"}

    a.raw_data = NIFD_DATA
    a.knn_data = NIFD_DATA_KNN

    a.id_column = 'subj_id'
    a.filter_duplicates_by = a.id_column
    a.site = "SITE"


class NIFDDataLoader(DataLoader):

    def __init__(self):
        super().__init__(specifier=NIFD_SPECIFICATIONS)
