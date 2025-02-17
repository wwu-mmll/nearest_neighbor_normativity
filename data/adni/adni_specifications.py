from shared import DATA_FOLDER, DataLoader, Specifications

ADNI_DATA = DATA_FOLDER.joinpath('adni/adni_joined.csv')
ADNI_KNN_RESULTS_DATA = DATA_FOLDER.joinpath('adni/adni_knn.csv')

ADNI_SPECIFICATIONS = Specifications('ADNI')
with ADNI_SPECIFICATIONS as a:
    a.age_column = 'age'
    a.age_needs_rounding = True

    a.female_identifier = 2
    a.sex_column = 'PTGENDER'

    a.filter_duplicates_by = "RID"

    a.diagnosis_column = 'diagnosis'
    a.healthy_identifier = 'CN'
    a.disease_dict = {"AD": "AD", "MCI": "MCI"}

    a.raw_data = ADNI_DATA
    a.knn_data = ADNI_KNN_RESULTS_DATA

    a.id_column = 'RID'
    a.site = "SITEID"


class ADNIDataLoader(DataLoader):

    def __init__(self):
        super().__init__(specifier=ADNI_SPECIFICATIONS)
