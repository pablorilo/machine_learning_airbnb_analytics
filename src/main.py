import pandas as pd

from data import AnalysisDate
from preprocessing import PreprocessingDate
print('hola')
if __name__ == '__main__':
    def run():
        
        # -- 0: Variables de control
        show_more_info = True
        test = False
        airbnb = pd.read_csv("C:/Users/prilo/OneDrive/Desktop/KeepCoding/proyectos/machine_learning_airbnb_analytics/data/raw/airbnb-listings-extract.csv", sep=";")

        analysis_dict = AnalysisDate(airbnb, show_more_info = show_more_info).analize()
        train_df = analysis_dict["train"]
        test_df = analysis_dict["test"]

        preprocesing_dict = PreprocessingDate(train_df).run()
        
    run()