
from sklearn.model_selection import train_test_split
import pandas as pd

# Importamos nuestro modulo para graficar
from graphics import Graphics

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

class AnalysisDate(Graphics):
    def __init__(self, df: pd.DataFrame, show_more_info: bool = True):
        #Instanciamos la superclase de Graphics para tener acceso desde la clase 
        super().__init__()
        self.df = df
        self.show_more_info = show_more_info

    def analize(self) -> dict:
        #1--Realizamos el filtrado de los alojamientos de Madrid
        self.printText('Analysis 1: Realizamos el filtrado del data set en bruto obteniendo únicamente los alojamientos de Madrid')
        print('[INFO] Realizando filtrado...')
        self.df = self.madridOnly(df=self.df)
        
        #2--Realizamos el split de los datos en train y test
        self.printText('Analysis 2: Realizamos la división en train y test del conjunto de datos')
        print('[INFO] Realizando división...\n')
        train, test = self.splitDataFrame()
        print(f'Dimension del grupo de train {train.shape}\n')
        print(f'Dimension del grupo de test {test.shape}\n')
        
        #3--Comprobamos como se distribuye la variable buscada de forma original, su raiz cuadrada y logaritmica
        self.printText('Analysis 3: Vamos a ver como se distribuye la variable buscada mediante gráficas de densidad de probabilidad')
        self.createAndSaveTargetDistribution(df = train, file_name = 'pricehistogram02.png')

        #4--Realizamos un análisis exploratorio de los datos
        self.printText('Analysis 4: Vamos a ver que tipos de datos tenemos y un poco de información sobre ellos')

        if self.show_more_info:
            print('<<<<<<<<<<<<<<<<<<<<<<<<<    Información general de dataFrame     >>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
            print(train.info())
            print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Resumen datos numéricos >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
            print(train.describe().T)
            print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  Resumen valores nulos  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
            print(train.isnull().sum())
            
        return {"train": train, "test": test}

    def madridOnly(self, df: pd.DataFrame, city: str= 'madrid') -> pd.DataFrame:
        """Funcion que recibe un dataframe en bruto y devuelve df con unicamente alojamientos de Madrid"""
        df['City'] = df['City'].fillna('')
        return df[df.City.str.contains(city, case=False)]

    def splitDataFrame(self, test_size: float = 0.2, shuffle: bool = True, random_state: int = 0):
        return train_test_split(self.df, test_size=test_size, shuffle=shuffle, random_state=random_state)
        