import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import math

class Graphics:
    def __init__(self, img_path : str = "./analysisImg/"):
        self.img_path = img_path
        os.makedirs(self.img_path, exist_ok=True) 

    def createAndSaveTargetDistribution(self, df: pd.DataFrame, file_name: str, target_col: str = 'Price') -> None:
        """Crea una imagen con las gráficas de distribucion de la etiqueta pasada como parámetro y la guarda en la ruta img_path
        :param df: Dataframe
        :param target_col_name: Nombre de la columna objetivo
        :param file_name: Nombre y extension del archivo
        :return: None """

        if not os.path.exists(f'{self.img_path}{file_name}'):
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 6))
            #Realizamos la grafica de distribución de la etiqueta
            sns.distplot(
                        df[target_col],
                        hist    = False,
                        rug     = True,
                        color   = "blue",
                        kde_kws = {'fill': True, 'linewidth': 1},
                        ax      = axes[0]
                    )
            axes[0].set_title("Distribución original", fontsize = 'medium')
            axes[0].set_xlabel('{target_col}', fontsize='small') 
            axes[0].tick_params(labelsize = 6)

            sns.distplot(
                np.sqrt(df[target_col]),
                hist    = False,
                rug     = True,
                color   = "blue",
                kde_kws = {'fill': True, 'linewidth': 1},
                ax      = axes[1]
            )
            axes[1].set_title("Transformación raíz cuadrada", fontsize = 'medium')
            axes[1].set_xlabel('sqrt({target_col})', fontsize='small') 
            axes[1].tick_params(labelsize = 6)

            sns.distplot(
                np.log(df[target_col]),
                hist    = False,
                rug     = True,
                color   = "blue",
                kde_kws = {'fill': True, 'linewidth': 1},
                ax      = axes[2]
            )
            axes[2].set_title("Transformación logarítmica", fontsize = 'medium')
            axes[2].set_xlabel(f'log({target_col})', fontsize='small') 
            axes[2].tick_params(labelsize = 6)

            fig.tight_layout()

            fig.savefig(f"{self.img_path}{file_name}", bbox_inches="tight")
            print(f"{self.img_path}{file_name} creado correctamente")

        else:
            print(f"La imagen {file_name} ya existe")

    def createAndSaveHistogram(self, df: pd.DataFrame, file_name : str, target_col: str= 'Price', columns:list = None) -> None:
        """ Crea una imagen con los histogramas en subplots y la guarda en la ruta img_path. la función calcula el numero de filas a generar y extrae las columnas numéricas del dataframe por defecto
        Si solo se quiere graficar ciertas dimensiones se puede pasar por parametro en forma de lista:
        :param df: Dataframe
        :param file_name: Nombre y extension del archivo
        :param target_col: nombre de la columna etiqueta, por defecto el Price
        :return: None """ 
        if not os.path.exists(f'{self.img_path}{file_name}'):
               
            num_columns= df[columns].colums if columns else df.select_dtypes(include=['float64', 'int64']).columns.drop(target_col)
            size= math.ceil(len(num_columns)/3)
            fig, axes = plt.subplots(nrows=size, ncols=3, figsize=(18, 5*size))
            axes = axes.flat
            fig.suptitle('Distribución variables numéricas',va='top', y= 1,fontsize = 18)
            for i, col in enumerate(num_columns):
                sns.histplot(
                    data    = df,
                    x       = col,
                    stat    = "count",
                    kde     = True,
                    color   = (list(plt.rcParams['axes.prop_cycle'])*5)[i]["color"],
                    line_kws= {'linewidth': 2},
                    alpha   = 0.3,
                    ax      = axes[i]
                )
                axes[i].set_title(col, fontsize = 12)
                axes[i].tick_params(labelsize = 10)
                plt.subplots_adjust(top=0.9)
                axes[i].set_xlabel("")
                fig.tight_layout()
            #Borramos los subplots que queden vacios
            graf_del = size*3 - len(num_columns)
            if graf_del == 1 :
                fig.delaxes(axes[-1])
            if graf_del == 2 :
                fig.delaxes(axes[-1])
                fig.delaxes(axes[-2])
            fig.savefig(f"{self.img_path}{file_name}", bbox_inches="tight")
            print(f"{self.img_path}{file_name} creado correctamente")

        else:
            print(f"La imagen {file_name} ya existe")

    def createAndSaveScatter(self, df: pd.DataFrame, file_name : str, target_col: str= 'Price', columns:list = None):
        """ Crea una imagen con los scatters en subplots y la guarda en la ruta img_path. la función calcula el numero de filas a generar y extrae las columnas numéricas del dataframe por defecto
        Si solo se quiere graficar ciertas dimensiones se puede pasar por parametro en forma de lista:
        :param df: Dataframe
        :param file_name: Nombre y extension del archivo
        :param target_col: nombre de la columna etiqueta, por defecto el Price
        :param columns: por defecto None, lo cual implica que grafica todos las dimensiones del df, 
                        se puede pasar una lista con las columnas que se desea graficar """
        if not os.path.exists(f'{self.img_path}{file_name}'):
            num_columns= df[columns].colums if columns else df.select_dtypes(include=['float64', 'int64']).columns.drop(target_col)
            size= math.ceil(len(num_columns)/3)
            fig, axes = plt.subplots(nrows=size, ncols=3, figsize=(18, 5*size))
            axes = axes.flat
            fig.suptitle('Distribución variables numéricas',va='top', y= 1,fontsize = 18)
            for i, col in enumerate(num_columns):
                sns.histplot(
                    data    = df,
                    x       = col,
                    stat    = "count",
                    kde     = True,
                    color   = (list(plt.rcParams['axes.prop_cycle'])*5)[i]["color"],
                    line_kws= {'linewidth': 2},
                    alpha   = 0.3,
                    ax      = axes[i]
                )
                axes[i].set_title(col, fontsize = 12)
                axes[i].tick_params(labelsize = 10)
                plt.subplots_adjust(top=0.9)
                axes[i].set_xlabel("")
                fig.tight_layout()
            #Borramos los subplots que queden vacios
            graf_del = size*3 - len(num_columns)
            if graf_del == 1 :
                fig.delaxes(axes[-1])
            if graf_del == 2 :
                fig.delaxes(axes[-1])
                fig.delaxes(axes[-2]) 
            fig.savefig(f"{self.img_path}{file_name}", bbox_inches="tight")
            print(f"{self.img_path}{file_name} creado correctamente")

        else:
            print(f"La imagen {file_name} ya existe")

    def createAndSaveBoxPlot(self, df: pd.DataFrame, file_name : str, target_col: str= 'Price', columns:list = None):
        """Crea graficos boxplot de todos las dimensiones de df con tipo ['object', 'uint8'] y las guarda en la ruta img_path, si se le pasa una lista con 
            dimensiones del df unicamente graficará dicha lista si se le pasa columns por parámetro graficará dichas columnas en base al target_col
        :param df: Dataframe
        :param file_name: Nombre y extension del archivo
        :param target_col: nombre de la columna etiqueta, por defecto el Price
        :param columns: por defecto None, lo cual implica que grafica todos las dimensiones del df, 
                        se puede pasar una lista con las columnas que se desea graficar """ 
        if not os.path.exists(f'{self.img_path}{file_name}'):
            object_columns= df[columns].columns if columns else df.select_dtypes(include=['object', 'uint8']).columns
    #Para realizar una buena visualizacion de los dummies vamos a eliminar del conteo para las filas los dummies que representan los SI
            columns_to_drop = [col for col in object_columns if ("_NO") in col]  
            object_columns = object_columns.drop(columns_to_drop) if columns_to_drop else  object_columns
            size= math.ceil(len(object_columns)/2)
            fig, axes= plt.subplots(nrows=size, ncols=2, figsize=(16, 6*size))
            axes = axes.flat
            fig.suptitle('Distribución del precio por variable categórica',va='top', y= 1,fontsize = 18)
            for i, col in enumerate(object_columns):
                if df[col].dtypes == 'object':  
                    sns.boxplot(data= df,
                                x           = 'Price',
                                y           = col,
                                width       = 0.7,
                                ax          = axes[i]
                        )
                    axes[i].set_title(f"{col}", fontsize = 12)
                    axes[i].tick_params(labelsize = 10)
                    axes[i].set_xlabel("")
                    axes[i].set_ylabel("")    
                    
                if df[col].dtypes == 'uint8':            
                    sns.boxplot(data= df,
                                x           = col,
                                y           = 'Price',
                                width       = 0.7,
                                ax          = axes[i]
                        )
                    axes[i].set_title(f"{col[:-3]} (NO=0 SI=1)", fontsize = 12)
                    axes[i].tick_params(labelsize = 10)
                    axes[i].set_xlabel("")
                    axes[i].set_ylabel("")               
            
            fig.tight_layout()
            #Borramos los subplots que queden vacios
            graf_del = size*2 - len(object_columns)
            if graf_del == 1 :
                fig.delaxes(axes[-1])
            fig.savefig(f"{self.img_path}{file_name}", bbox_inches="tight")
            print(f"{self.img_path}{file_name} creado correctamente")
        else:
            print(f"La imagen {file_name} ya existe")

    def createAndSaveCategoricalDistribution(self, df: pd.DataFrame, file_name : str, columns:list = None):
        """Crea gráficos de como se distribuyen los datos en las variables categóricas y lo guarda como imagen
        :param df: Dataframe
        :param file_name: Nombre y extension del archivo
        :param columns: por defecto None, lo cual implica que grafica todos las dimensiones del df, 
                        se puede pasar una lista con las columnas que se desea graficar """
        if not os.path.exists(f'{self.img_path}{file_name}'):
            object_columns= df[columns].columns if columns else df.select_dtypes(include=['object']).columns
            num_plots = len(object_columns)
            num_rows, num_cols = divmod(num_plots, 2)
            if num_cols != 0:
                num_rows += 1
            fig, axes= plt.subplots(nrows=num_rows, ncols=2, figsize=(16, 6*num_rows))
            axes = axes.flat
            columnas_object = df.select_dtypes(include=['object']).columns
            fig.suptitle('Distribución variables categóricas',fontsize = 18)
            for i, colum in enumerate(columnas_object):
                df[colum].value_counts().plot.barh(ax = axes[i])
                axes[i].set_title(colum, fontsize = 12)
                axes[i].tick_params(labelsize = 10)
                axes[i].set_xlabel("")

            # Se eliminan los axes vacíos
            graf_del = num_rows*2 - len(object_columns)
            if graf_del == 1 :
                fig.delaxes(axes[-1])

            fig.tight_layout()
            plt.subplots_adjust(top=0.9)
            fig.savefig(f"{self.img_path}{file_name}", bbox_inches="tight") 
            print(f"{self.img_path}{file_name} creado correctamente")  
        else:
            print(f"La imagen {file_name} ya existe")

    def createAndSaveCorrelationMatrix(self, df: pd.DataFrame, file_name : str, target_col: str= 'Price'):
        """Crea la matriz de correlación de las dimensiones numéricas del df
            :param df : Dataframe
            :param file_name: Nombre y extension del archivo
            """
        if not os.path.exists(f'{self.img_path}{file_name}'):
            num_colums = df.select_dtypes(include=['float64', 'int64'])
            corr = np.abs(num_colums.drop([target_col], axis=1).corr())

            # Generate a mask for the upper triangle
            mask = np.zeros_like(corr, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True

            # Set up the matplotlib figure
            fig, ax = plt.subplots(figsize=(8, 8))

            # Draw the heatmap with the mask and correct aspect ratio
            sns.heatmap(corr, mask=mask,vmin = 0.0, vmax=1.0, center=0.5, annot = True,
                        linewidths=.1, cmap="YlGnBu", cbar_kws={"shrink": .8})
            fig.suptitle('Matriz de correlación', fontsize=18)
            fig.savefig(f"{self.img_path}{file_name}", bbox_inches="tight")  
            print(f"{self.img_path}{file_name} creado correctamente") 
        else:
            print(f"La imagen {file_name} ya existe")

    def createAndSaveplotGridValues(self, vector, scores,file_name: str, metric: str, n_folds: int, x_label: str):
        if not os.path.exists(f'{self.img_path}{file_name}'):
            fig, ax = plt.subplots()
            ax.semilogx(vector,scores,'-o')
            plt.xlabel(x_label,fontsize=16)
            plt.ylabel(f'{n_folds}-Fold MSE')
            plt.savefig(f"{self.img_path}{file_name}")
            print(f"{self.img_path}{file_name} creado correctamente")
        else:
            print(f"La imagen {file_name} ya existe")

    @staticmethod
    def printText(text_to_print: str):
        print(f'\n#######################################################################################\n'
              f'----    {text_to_print}\n'
              f'#######################################################################################\n')



airbnb = pd.read_csv("data/raw/airbnb-listings-extract.csv", sep=";")

prueba = Graphics()
#prueba.createAndSaveTargetDistribution(df = airbnb, target_col = 'Price', file_name='Prueba2.png')
#prueba.createAndSaveHistogram(df= airbnb, target_col='Price' ,file_name='prueba3.png')
#prueba.createAndSaveScatter(df= airbnb, target_col='Price' ,file_name='prueba4.png')
prueba.createAndSaveCategoricalDistribution(df= airbnb, file_name='prueba6.png', columns=['Name','City','Room Type'])

airbnb.info()