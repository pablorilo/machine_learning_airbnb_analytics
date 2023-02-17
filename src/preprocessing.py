import numpy as np
import pandas as pd

from feature_engine.encoding import MeanEncoder

from sklearn.preprocessing import StandardScaler

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
            colums_to_remove =['ID','Host ID','Listing Url','Scrape ID','Last Scraped','Neighbourhood Cleansed','Name','Summary','Space','Description','Experiences Offered','Neighborhood Overview','Notes','Transit',
                    'Access','Interaction','House Rules','Thumbnail Url','Medium Url','Picture Url','XL Picture Url','Host URL','Host Name','Host About', 'Host Response Rate',
                    'Calculated host listings count','Host Thumbnail Url','Host Picture Url','Host Verifications','Features','Square Feet','Host Acceptance Rate','Has Availability',
                    'Jurisdiction Names','Latitude','Longitude','Weekly Price','Monthly Price','Country','License','Geolocation','Calendar last Scraped','Calendar Updated','Square Feet',
                    'Host Neighbourhood','Host Response Time','Reviews per Month','First Review','Last Review','Review Scores Rating','Review Scores Accuracy','Review Scores Cleanliness',
                    'Review Scores Checkin','Review Scores Communication','Review Scores Location','Review Scores Value','Reviews per Month','Neighbourhood','Market','State','Zipcode',
                    'Smart Location','Country Code','Host Location','Host Listings Count','Host Total Listings Count','Host Since','City','Street','Guests Included','Number of Reviews',
                    'Minimum Nights', 'Maximum Nights', 'Availability 30', 'Availability 60', 'Availability 90', 'Availability 365','Cleaning Fee'] 
            self.df.drop(colums_to_remove, axis=1, inplace=True)
            self.dummiesAndCategoricalTransformations()
            
        else:
            #1--Indicamos los datos a cargar
            self.printText(f'Preprocessing 1: Cargamos los datos de {self.group} para realizar el preprocesado del mismo')
            print('[INFO] Cargando datos...')
            #2--Eliminamos del df las columnas no relevantes
            self.printText(f'Preprocessing 2: Eliminamos las columnas no relevantes, como url, fotos, descripciones... de df del grupo de {self.group}.')
            print('[INFO] Eliminado columnas..\n.')
            colums_to_remove = ['ID','Host ID','Listing Url','Scrape ID','Last Scraped','Neighbourhood Cleansed','Name','Summary','Space','Description','Experiences Offered','Neighborhood Overview','Notes','Transit',
                'Access','Interaction','House Rules','Thumbnail Url','Medium Url','Picture Url','XL Picture Url','Host URL','Host Name','Host About', 'Host Response Rate',
                'Calculated host listings count','Host Thumbnail Url','Host Picture Url','Host Verifications','Features','Square Feet','Host Acceptance Rate','Has Availability',
                'Jurisdiction Names','Latitude','Longitude','Weekly Price','Monthly Price','Country','License','Geolocation','Calendar last Scraped','Calendar Updated','Square Feet',
                'Host Neighbourhood','Host Response Time','Reviews per Month','First Review','Last Review','Review Scores Rating','Review Scores Accuracy','Review Scores Cleanliness',
                'Review Scores Checkin','Review Scores Communication','Review Scores Location','Review Scores Value','Reviews per Month','Neighbourhood','Market','State','Zipcode',
                'Smart Location','Country Code','Host Location','Host Listings Count','Host Total Listings Count','Host Since','City','Street','Guests Included','Number of Reviews']


            self.df.drop(colums_to_remove, axis=1, inplace=True)
            print('<<<<<<<<<<<<<<<<<< Visualizamos nuevamente los valores nulos. >>>>>>>>>>>>>>>>>>\n')
            print(self.df.isnull().sum())

            #3--Variables Numéricas. Mediante la clase Graphics generamos diferentes graficos de distribución de las variables 
            ##Graficamos##
            self.printText('Preprocesing 3: Estudio de variables numéricas.')
            print('\n-----------------------------------------------------------------------------------------\n')
            print('Visualizamos su distribución\n')
            self.createAndSaveHistogram(df = self.df, file_name='numerichistogram2.png')
            print('\n-----------------------------------------------------------------------------------------\n')
            print('Visualizamos gráficos scatter respecto a la variable objetivo\n')
            self.createAndSaveScatter(df = self.df, file_name='numericscatter2.png')
            print('\n-----------------------------------------------------------------------------------------\n')
            print('Visualizamos matriz de correlacion de todas las dimensiones\n')
            self.createAndSaveCorrelationMatrix(df = self.df, file_name='correlationmatrix2.png')  
            print('\n-----------------------------------------------------------------------------------------\n') 
            print('Tras visualizar los scatterplot observamos que hay variables que a priori se relacionan bastante con el precio, como pueden ser Accommodates, Bathrooms, Bedrooms, beds, Cleaning Fee y Extra People. En las siguientes celdas vamos a analizar un poco más a fondo las dos últimas dimensiones mencionadas. Además también observamos que Minimum Nights y Maximum Nights en principio aportarían poco al modelo por lo tanto las vamos a eliminar. Además vamos a eliminar las dimensiones de Availability por que tampoco parece que tengan una fuerte relación con el precio y están muy correladas entre si')     
            print('\Visualizamos la grafica Scatter de como se correlacionan los Availability\n')
            self.createAndSaveScatter(df = self.df, file_name='numericscatter3.png',target_col = 'Availability 30', columns=['Availability 60'])
            self.df.drop(['Minimum Nights', 'Maximum Nights', 'Availability 30', 'Availability 60', 'Availability 90', 'Availability 365' ],axis=1, inplace=True)
            print('\n-----------------------------------------------------------------------------------------\n')
            #Vamos a realizar un estudio sobre Cleaning Fee,vamos asumir que los alojamientos que no tienen asignado un valor de Cleaning Fee es por que no se cobra dicha tarifa, por lo que los pondremos a cero
            print('Vamos asumir que los alojamientos que no tienen asignado un valor de Cleaning Fee es por que no se cobra dicha tarifa, por lo que los pondremos a cero. Ademas el Security Deposit, lo trasnformaremos a categórico donde 0 es que no se requiere depósito y 1 que si. Por otro lado para saber si el alojamiento tiene aire acondicionado y calefacción creamos dos columnas en donde indica (SI/NO). Por último convertiremos a dummies las 3 columnas trasnformadas.')
            #Analizamos la variable Cleaning Fee
            print('-----------------------------------------------------------------------------')
            self.df['Cleaning Fee'] = self.df['Cleaning Fee'].fillna(0)
            self.dummiesAndCategoricalTransformations()
            print('Resumen de la variable "Cleaning Fee":')
            print(self.df['Cleaning Fee'].describe())
            print('-----------------------------------------------------------------------------')
            filtered = (self.df['Cleaning Fee'][self.df['Cleaning Fee']>100])
            print(f'Existen {filtered.shape[0]} valores de Cleaning Fee superiores a 100 euros, distribuidos de la siguiente manera:')
            print(filtered.value_counts())
            print('-----------------------------------------------------------------------------')
            print(self.df[self.df['Cleaning Fee']>100].head(15))
            print('\nVamos a optar por la eliminación de la variable Cleaning Fee, ya que existen bastantes valores elevados y que analizando un poco el dataser filtrado parece que éste importe tiene bastante relación con el mínimo de noches de alojamiento.\n')
            self.df.drop(['Cleaning Fee'], axis=1, inplace=True)

            #Analizamos la variable Extra People
            print('-----------------------------------------------------------------------------')
            print('Resumen de la variable "Extra People":')
            print('Resumen de la variable:')
            print(self.df['Extra People'].describe())
            print('-----------------------------------------------------------------------------')
            filtered = (self.df['Extra People'][self.df['Extra People']>60])
            print(f'Existen {filtered.shape[0]} valores de Cleaning Fee superiores a 100 euros, distribuidos de la siguiente manera:')
            print(filtered.value_counts())
            print('-----------------------------------------------------------------------------')
            self.df[self.df['Extra People']>30].head(15)
            print('\nVamos a optar por eliminar tambiés esta variable por que a priori no parece muy fiable, ya que tiene valores de extrapeople muy altos para alojamientos muy pequeños .\n')
            self.df.drop(['Extra People'], axis=1, inplace=True)

            print('-----------------------------------------------------------------------------')
            print('Resumen de las variables que nos quedan en el dataframe')
            print(self.df.info())

            #4--Variables Categoricas. Mediante la clase Graphics generamos diferentes graficos de distribución de las variables 
            ##Graficamos##
            self.printText('Preprocesing 4: Estudio de variables categóricas.')
            print('Resumen de las variables tipo object\n')
            print(self.df.select_dtypes(include=['object']).describe())
            print('\n-----------------------------------------------------------------------------------------\n')
            print('Visualizamos su distribución\n')
            #self.createAndSaveCategoricalDistribution(df = self.df, file_name='categoricalhistogram2.png')
            print('\n-----------------------------------------------------------------------------------------\n')
            print('Vamos a comprobar cuantos valores tienen los diferentes niveles de Cancelation Policy\n')
            print(self.df['Cancellation Policy'].value_counts())
            print('\n Reagrupamos los niveles que contienen sctric en uno solo.\n')
            list_index_canc_Pol = self.df['Cancellation Policy'].value_counts().index[0:3] 
            self.regroupLevels(column='Cancellation Policy',list_index = list_index_canc_Pol, name_new_group='strict')
            print('\n Comprobamos la nueva reagrupación:\n')
            print(self.df['Cancellation Policy'].value_counts())
            print('\n-----------------------------------------------------------------------------------------\n')
            print('Ahora vamos a realizar lo mismo con los niveles de Property Type\n')
            print(self.df['Property Type'].value_counts())
            print('\n Reagrupamos los niveles manteniendo hasta loft y el resto los tipificamos como Other.\n')
            list_index_prop = self.df['Property Type'].value_counts().index[0:5] 
            self.regroupLevels(column='Property Type',list_index = list_index_prop)
            print('\n Comprobamos la nueva reagrupación:\n')
            print(self.df['Property Type'].value_counts())
            print('\n-----------------------------------------------------------------------------------------\n')
            print('Ahora es el turno de Bed Type\n')
            print(self.df['Bed Type'].value_counts())
            print('\n Reagrupamos los niveles manteniendo hasta loft y el resto los tipificamos como Other.\n')
            list_index_Bed = self.df['Bed Type'].value_counts().index[0:2] 
            self.regroupLevels(column='Bed Type',list_index = list_index_Bed)
            print('\n Comprobamos la nueva reagrupación:\n')
            print(self.df['Bed Type'].value_counts())
            print('\n-----------------------------------------------------------------------------------------\n')
            print('Visualizamos como se distribuye en la variable objetivo por categorias')
            self.createAndSaveBoxPlot(df=self.df,file_name='boxplot1.png')
            print('\n-----------------------------------------------------------------------------------------\n')
            print('Ahora vamos a realizar la trasnformación de las variables '"Neighbourhood_Group_Cleansed"','"Property_Type"','"Room_Type"', '"Bed_Type"', '"Cancellation_Policy"', mediante MeanEncoder de la libreria Feature_engine.\n Vamos a visualizar que tipo de datos tenemos en este moment:\n')
            print(self.df.info())
            columns_to_mean =['Neighbourhood_Group_Cleansed','Property_Type','Room_Type', 'Bed_Type', 'Cancellation_Policy']
            x_train, y_train = self.meanEncoder(columns_to_mean= columns_to_mean )
            max_price = max(max_price)
            print('\n-----------------------------------------------------------------------------------------\n')
            print('Ahora vamos a volver a visualizar los datos en gráficos')
            print('Visualizamos gráficos boxplot de las variables categoricas\n')
            self.createAndSaveBoxPlot(df=self.df,file_name='boxplot2.png')
            print('\n-----------------------------------------------------------------------------------------\n')
            print('Visualizamos gráficos scatter respecto a la variable objetivo\n')
            self.createAndSaveScatter(df = self.df, file_name='numericscatter2.png')
            print('\n-----------------------------------------------------------------------------------------\n')
            print('Visualizamos matriz de correlacion de todas las dimensiones\n')
            self.createAndSaveCorrelationMatrix(df = self.df, file_name='correlationmatrix2.png') 
            print('\n-----------------------------------------------------------------------------------------\n')

            #5--Escalado de los datos. Mediante la libreria StandarScaler de sklearn 
            
            self.printText('Preprocesing 3: Escalado de los datos.')
            print('\n-----------------------------------------------------------------------------------------\n')
            
            return {'df': self.df}
            
    def scalerDate(self, x:pd.DataFrame, y: pd.DataFrame):
        pass

    def meanEncoder(self, columns_to_mean:list, target_col: str = 'Price'):
        """Función que realiza la codificación de las variables categoricas con la media de la variable buscada y devuelve el set ya dividio en x \
                e y
             :param columns_to_mean: lista de columnas a codificar
             :param target_col: columna objetivo sobre la q se realiza la media """
        print('[INFO] Realizando el codificado...')
        self.df.dropna(inplace= True)
        self.df.columns = self.df.columns.str.replace(' ', '_')
        y_train = self.df[target_col]
        x_train = self.df.drop(target_col,axis =1)
        x_train_names_col = self.df.columns
        #Creamos la instancia de meanEncoder
        main_encoder_categorical = MeanEncoder(variables=columns_to_mean).fit(x_train, y_train)
        x_train = main_encoder_categorical.transform(x_train)
        return x_train, y_train        



    def regroupLevels(self,  column: str, list_index: list,name_new_group: str = 'Other',):
        """Esta función realiza una reagrupación de niveles de una columna de variable categórica:
                df = dataframe de pandas
                column = columna a modificar 
                list_index = lista de valores que se mantendrán, al resto que no se correspondan con la lista les pondrá el valor other por defecto
                new_value = 'Other' valor que añadirá por defecto"""
        print('[INFO] Realizando la reagrupación...')
        self.df[column] = [name_new_group if x not in list_index else x for x in self.df[column]]

    
    def dummiesAndCategoricalTransformations(self):
        """crea las dummies tras trasnformar a categóricas 3 dimensiones, la existencia de A/C, la existencia calfaccción en la vivienda y 
        si se cobra o no fianza. Por último elimina la columna Amenities"""
        print('[INFO] Realizando las transformaciones...')
        self.df['Security Deposit'] = self.df['Security Deposit'].map(lambda x: 'SI' if x>0 else 'NO')
        condition_heat = self.df['Amenities'].str.contains('heating', case=False)
        condition_air = self.df['Amenities'].str.contains('Air conditioning', case=False)
        self.df['Heating'] = np.where(condition_heat, 'SI','NO')
        self.df['Air conditioning'] = np.where(condition_air, 'SI','NO')
        #Trasnforma las 3 columnas anteriores a dummies
        self.df = pd.get_dummies(self.df,prefix=['A/C','Heat','Sec_Dep'],columns=['Air conditioning','Heating','Security Deposit'])
        return self.df.drop('Amenities', axis=1)
        