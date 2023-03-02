import pandas as pd

from data import AnalysisDate
from preprocessing import PreprocessingDate
from modeling import Modeling


if __name__ == '__main__':
    def run():        
        # -- 0: Variables de control
        show_more_info = True
        test = False
        airbnb = pd.read_csv("C:/Users/prilo/OneDrive/Desktop/KeepCoding/proyectos/machine_learning_airbnb_analytics/data/raw/airbnb-listings-extract.csv", sep=";")

        analysis_dict = AnalysisDate(airbnb, show_more_info = show_more_info).analize()
        train_df = analysis_dict["train"]
        test_df = analysis_dict["test"]

        preprocesing = PreprocessingDate(train_df)
        preprocesing_dict_train = preprocesing.run(train_df)
        preprocesing_dict_test = preprocesing.run(test_df,test=True)
        data_dict = {}
        data_dict.update(preprocesing_dict_train)
        data_dict.update(preprocesing_dict_test)
        
        Modeling(data_dict=data_dict).run(method='LINEAR')
        Modeling(data_dict=data_dict).run(method='RIDGE')
        Modeling(data_dict=data_dict).run(method='LASSO')
        Modeling(data_dict=data_dict).run(method='FOREST')





    run()