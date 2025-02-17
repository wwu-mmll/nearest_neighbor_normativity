from data import *
from model.knn_projector import KNNProjection
from ml.knn_brainage import apply_svm_brain_age
from normative.train_model import NormativeModel


if __name__ == '__main__':

    for dl in [
               FOR2107DataLoader,               
               OASISDataLoader,
               ADNIDataLoader,
               AIBLDataLoader,
               NIFDDataLoader               
               ]:
        data_loader = dl()
        data_loader.load_data(data_spec='raw_data')

        knn_projector = KNNProjection(data_loader=data_loader)
        knn_projector.run()
        apply_svm_brain_age(data_loader)

        # trainer = NormativeModel()
        # trainer.predict(data_loader)







