from shared import DATA_FOLDER, DataLoader, Specifications

FOR_DATA = DATA_FOLDER.joinpath('for2107/for2107_joined.csv')
FOR_KNN_DATA = DATA_FOLDER.joinpath('for2107/for2107_knn.csv')

FOR2107_SPECIFICATIONS = Specifications('FOR2107')

with FOR2107_SPECIFICATIONS as a:
    a.age_column = 'Alter'

    a.female_identifier = 2
    a.sex_column = 'Geschlecht'

    a.diagnosis_column = 'Group'
    a.healthy_identifier = 1
    a.disease_dict = {"MDD": 2, "BD": 3, "SCZ": 5, "SCZAFF": 4, "ANX": 7}

    # a.site = "Site"
    a.filter_duplicates_by = "id"

    a.raw_data = FOR_DATA
    a.knn_data = FOR_KNN_DATA

    a.id_column = 'id'



class FOR2107DataLoader(DataLoader):

    def __init__(self):
        super().__init__(specifier=FOR2107_SPECIFICATIONS)

    def apply_filter(self, data_to_filter):
        return data_to_filter
