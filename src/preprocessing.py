import numpy as np
import pandas as pd
from sklearn import preprocessing

# Importamos nuestro modeulo para graficar
from graphics import Graphics

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


class PreprocessingDate(Graphics):
    def __init__(self, df: pd.DataFrame, test = False , show_more_info: bool = True):
        #Instanciamos la superclase de Graphics para tener acceso desde la clase 
        super().__init__()
        self.df = df
        self.show_more_info = show_more_info
        if test:
            self.group = 'test'
        else:
            self.group = 'train'

    def run(self, test = False)-> dict:
        if test:
            colums_to_remove = ['ID','Host ID','Listing Url','Scrape ID','Last Scraped','Neighbourhood Cleansed','Name','Summary','Space','Description','Experiences Offered','Neighborhood Overview','Notes','Transit',
                'Access','Interaction','House Rules','Thumbnail Url','Medium Url','Picture Url','XL Picture Url','Host URL','Host Name','Host About', 'Host Response Rate',
                'Calculated host listings count','Host Thumbnail Url','Host Picture Url','Host Verifications','Features','Square Feet','Host Acceptance Rate','Has Availability',
                'Jurisdiction Names','Latitude','Longitude','Weekly Price','Monthly Price','Country','License','Geolocation','Calendar last Scraped','Calendar Updated','Square Feet',
                'Host Neighbourhood','Host Response Time','Reviews per Month','First Review','Last Review','Review Scores Rating','Review Scores Accuracy','Review Scores Cleanliness',
                'Review Scores Checkin','Review Scores Communication','Review Scores Location','Review Scores Value','Reviews per Month','Neighbourhood','Market','State','Zipcode',
                'Smart Location','Country Code','Host Location','Host Listings Count','Host Total Listings Count','Cleaning Fee','Host Since','City','Street','Guests Included','Number of Reviews']

            self.df.drop(colums_to_remove, axis=1, inplace=True)
            self.dummiesAndCategoricalTransformations()
            
        else:
            self.printText(f'Preprocessing 1: Cargamos los datos de {self.group} para realizar el preprocesado del mismo')
            #1--Eliminamos del df las columnas no relevantes
            self.printText(f'Preprocessing 2: Eliminamos las columnas no relevantes, como url, fotos, descripciones... de df del grupo de {self.group}.')
            colums_to_remove = ['ID','Host ID','Listing Url','Scrape ID','Last Scraped','Neighbourhood Cleansed','Name','Summary','Space','Description','Experiences Offered','Neighborhood Overview','Notes','Transit',
                    'Access','Interaction','House Rules','Thumbnail Url','Medium Url','Picture Url','XL Picture Url','Host URL','Host Name','Host About', 'Host Response Rate',
                    'Calculated host listings count','Host Thumbnail Url','Host Picture Url','Host Verifications','Features','Square Feet','Host Acceptance Rate','Has Availability',
                    'Jurisdiction Names','Latitude','Longitude','Weekly Price','Monthly Price','Country','License','Geolocation','Calendar last Scraped','Calendar Updated','Square Feet',
                    'Host Neighbourhood','Host Response Time','Reviews per Month','First Review','Last Review','Review Scores Rating','Review Scores Accuracy','Review Scores Cleanliness',
                    'Review Scores Checkin','Review Scores Communication','Review Scores Location','Review Scores Value','Reviews per Month','Neighbourhood','Market','State','Zipcode',
                    'Smart Location','Country Code','Host Location','Host Listings Count','Host Total Listings Count','Host Since','City','Street','Guests Included','Number of Reviews',
                    'Minimum Nights', 'Maximum Nights', 'Availability 30', 'Availability 60', 'Availability 90', 'Availability 365']

            self.df.drop(colums_to_remove, axis=1, inplace=True)
            print('<<<<<<<<<<<<<<<<<< Visualizamos nuevamente los valores nulos. >>>>>>>>>>>>>>>>>>')
            print(self.df.isnull().sum())

            #2--Mediante la clase Graphics generamos diferentes graficos de distribución de las variables 
            print('\n-----------------------------------------------------------------------------------------\n')
            print('Visualizamos como se distribuyen las variables numéricas\n')
            self.createAndSaveHistogram(df = self.df, file_name='numerichistogram1.png')
            print('\n-----------------------------------------------------------------------------------------\n')
            print('Visualizamos gráficos scatter de las variables numéricas respecto a la variable objetivo\n')
            self.createAndSaveScatter(df = self.df, file_name='numericscatter1.png')
            print('\n-----------------------------------------------------------------------------------------\n')
            print('Visualizamos gráficos scatter de las variables numéricas respecto a la variable objetivo\n')
            self.createAndSaveCorrelationMatrix(df = self.df, file_name='correlationmatrix1.png')

            self.printText('Tras visualizar los scatterplot observamos que hay variables que a priori se relacionan bastante con el precio, como pueden ser Accommodates, Bathrooms,\
                           Bedrooms, beds, Cleaning Fee y Extra People. En las siguientes celdas vamos a analizar un poco más a fondo las dos últimas dimensiones \
                             mencionadas. Además también observamos que Minimum Nights y Maximum Nights en principio aportarían poco al modelo por lo tanto las vamos a eliminar.\
                             Además vamos a eliminar las dimensiones de Availability por que tampoco parece que tengan una fuerte relación con el precio y están muy correladas entre si')

            print('\Visualizamos la grafica Scatter de como se correlacionan los Availability\n')
            self.createAndSaveScatter(df = self.df, file_name='numericscatter1.png',target_col = 'Availability 30', columns=['Availability 60'])
            self.df.drop(['Minimum Nights', 'Maximum Nights', 'Availability 30', 'Availability 60', 'Availability 90', 'Availability 365' ], inplace=True)
            
            #3--Vamos a realizar un estudio sobre Cleaning Fee,vamos asumir que los alojamientos que no tienen asignado un valor de Cleaning Fee es por que no se cobra dicha tarifa, por lo que los pondremos a cero
            self.printText('Preprocessing 2: vamos asumir que los alojamientos que no tienen asignado un valor de Cleaning Fee es por que no se cobra dicha tarifa, \
                            por lo que los pondremos a cero. Ademas el Security Deposit, lo trasnformaremos a categórico donde 0 es que no se requiere depósito y 1 que si.\
                                 Por otro lado para saber si el alojamiento tiene aire acondicionado y calefacción creamos dos columnas en donde indica (SI/NO).\
                                     y por último convertiremos a dummies las 3 columnas trasnformadas.')
            self.df['Cleaning Fee'] = self.df['Cleaning Fee'].fillna(0)
            self.dummiesAndCategoricalTransformations()
            print('Resumen de la variable:')
            print(self.df['Cleaning Fee'].describe())
            print('-----------------------------------------------------------------------------')
            filtered = (self.df['Cleaning Fee'][self.df['Cleaning Fee']>100])
            print(f'Existen {filtered.shape[0]} valores de Cleaning Fee superiores a 100 euros, distribuidos de la siguiente manera:')
            print(filtered.value_counts())
            print('-----------------------------------------------------------------------------')
            print(self.df[self.df['Cleaning Fee']>100].head(15))
            print('\nVamos a optar por la eliminación de la variable Cleaning Fee, ya que existen bastantes valores \
                elevados y que analizando un poco el dataser filtrado parece que éste importe tiene bastante relación \
                    con el mínimo de noches de alojamiento.\n')
            self.df.drop(['Cleaning Fee'], axis=1, inplace=True)

            return {'df': self.df}

        def dummiesAndCategoricalTransformations(self):
            """crea las dummies tras trasnformar a categóricas 3 dimensiones, la existencia de A/C, la existencia calfaccción en la vivienda y 
                 si se cobra o no fianza. Por último elimina la columna Amenities"""
            self.df['Security Deposit'] = self.df['Security Deposit'].map(lambda x: 'SI' if x>0 else 'NO')
            condition_heat = self.df['Amenities'].str.contains('heating', case=False)
            condition_air = self.df['Amenities'].str.contains('Air conditioning', case=False)
            self.df['Heating'] = np.where(condition_heat, 'SI','NO')
            self.df['Air conditioning'] = np.where(condition_air, 'SI','NO')
            #Trasnforma las 3 columnas anteriores a dummies
            self.df = pd.get_dummies(self.df,prefix=['A/C','Heat','Sec_Dep'],columns=['Air conditioning','Heating','Security Deposit'])
            return self.df.drop('Amenities', axis=1)