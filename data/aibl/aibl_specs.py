from shared import DATA_FOLDER, DataLoader, Specifications

AIBL_DATA_FOLDER = DATA_FOLDER / "aibl"
AIBL_DATA = AIBL_DATA_FOLDER / "aibl_joined.csv"
AIBL_DATA_KNN = AIBL_DATA_FOLDER / "aibl_knn.csv"

AIBL_SPECIFICATIONS = Specifications('AIBL')
with AIBL_SPECIFICATIONS as a:
    a.age_column = 'AgeAtVisit'
    a.age_needs_rounding = True

    a.female_identifier = 2
    a.sex_column = 'PTGENDER'

    a.filter_duplicates_by = "RID"

    a.healthy_identifier = 0.    
    a.diagnosis_column = 'diagnosis'
    a.disease_dict = {"MCI": 0.5, "AD": 1}

    a.raw_data = AIBL_DATA
    a.knn_data = AIBL_DATA_KNN

    a.id_column = 'RID'
    a.site = 'SITEID_y'


class AIBLDataLoader(DataLoader):

    def __init__(self):
        super().__init__(specifier=AIBL_SPECIFICATIONS)
