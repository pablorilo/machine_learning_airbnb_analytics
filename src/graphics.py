import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
import pandas as pd
import numpy as np
import math

class Graphics:
    def __init__(self, img_path : str = "../analysisImg/"):
        self.img_path = img_path
        os.makedirs(self.img_path, exist_ok=True) 

    def createAndSaveTargetDistribution(self, df: pd.DataFrame, file_name: str, target_col: str = 'Price') -> None:
        """Crea una imagen con las gráficas de distribucion de la etiqueta pasada como parámetro y la guarda en la ruta img_path
        :param df: Dataframe
        :param target_col_name: Nombre de la columna objetivo
        :param file_name: Nombre y extension del archivo
        :return: None """
        print('[INFO] Creando imagen gráfica...')
        if not os.path.exists(f'{self.img_path}{file_name}'):
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 6))
            #Realizamos la grafica de distribución de la etiqueta
          
            # Realizamos la grafica de distribución de la etiqueta
            sns.kdeplot(
                df[target_col],
                color="blue",
                ax=axes[0],
            )
            axes[0].set_title("Distribución original", fontsize='medium')
            axes[0].set_xlabel(target_col, fontsize='small')
            axes[0].tick_params(labelsize=6)

            sns.kdeplot(
                np.sqrt(df[target_col]),
                color="blue",
                ax=axes[1],
            )
            axes[1].set_title("Transformación raíz cuadrada", fontsize='medium')
            axes[1].set_xlabel(f'sqrt({target_col})', fontsize='small')
            axes[1].tick_params(labelsize=6)

            sns.kdeplot(
                np.log(df[target_col]),
                color="blue",
                ax=axes[2],
            )
            axes[2].set_title("Transformación logarítmica", fontsize='medium')
            axes[2].set_xlabel(f'log({target_col})', fontsize='small')
            axes[2].tick_params(labelsize=6)

            fig.tight_layout()

            fig.savefig(f"{self.img_path}{file_name}", bbox_inches="tight")
            print(f"{self.img_path}{file_name} creado correctamente")

        else:
            print(f"[INFO] La imagen {file_name} ya existe")

    def createAndSaveHistogram(self, df: pd.DataFrame, file_name : str, target_col: str= 'Price', columns:list = None) -> None:
        """ Crea una imagen con los histogramas en subplots y la guarda en la ruta img_path. la función calcula el numero de filas a generar y extrae las columnas numéricas del dataframe por defecto
        Si solo se quiere graficar ciertas dimensiones se puede pasar por parametro en forma de lista:
        :param df: Dataframe
        :param file_name: Nombre y extension del archivo
        :param target_col: nombre de la columna etiqueta, por defecto el Price
        :return: None """ 
        print('[INFO] Creando imagen gráfica...')
        if not os.path.exists(f'{self.img_path}{file_name}'):
               
            num_columns= df[columns].columns if columns else df.select_dtypes(include=['float64', 'int64']).columns.drop(target_col)
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
            print(f"[INFO] {self.img_path}{file_name} creado correctamente")

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
        print('[INFO] Creando imagen gráfica...')
        if not os.path.exists(f'{self.img_path}{file_name}'):
            num_columns = df[columns].columns if columns else df.select_dtypes(include=['float64', 'int64']).columns.drop(target_col)
            size= math.ceil(len(num_columns)/3)
            fig, axes = plt.subplots(nrows=size, ncols=3, figsize=(16, 5*size))
            axes = axes.flat
            fig.suptitle('Relación entre variables',va='top', y= 1,fontsize = 18)
            for i, colum in enumerate(num_columns):
                sns.regplot(
                    x           = df[colum],
                    y           = df[target_col],
                    color       = "gray",
                    marker      = '.',
                    scatter_kws = {"alpha":0.4},
                    line_kws    = {"color":"r","alpha":0.5},
                    ax          = axes[i]
                )
                axes[i].set_title(f"{target_col} vs {colum}", fontsize = 12)
                axes[i].tick_params(labelsize = 10)
                axes[i].set_xlabel("")
                axes[i].set_ylabel("")
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
            print(f"[INFO] La imagen {file_name} ya existe")
   
    def createAndSaveCategoricalDistribution(self, df: pd.DataFrame, file_name : str, columns:list = None):
        mpl.rcParams['font.family'] = 'IPAexGothic'
        """Crea gráficos de como se distribuyen los datos en las variables categóricas y lo guarda como imagen
        :param df: Dataframe
        :param file_name: Nombre y extension del archivo
        :param columns: por defecto None, lo cual implica que grafica todos las dimensiones del df, 
                        se puede pasar una lista con las columnas que se desea graficar """
        print('[INFO] Creando imagen gráfica...')
        if not os.path.exists(f'{self.img_path}{file_name}'):
            print(columns)
            object_columns= df[columns].columns if columns else df.select_dtypes(include=['object']).columns
            num_plots = len(object_columns)
            print(num_plots)
            num_rows, num_cols = divmod(num_plots, 2)
            if num_cols != 0:
                num_rows += 1
            fig, axes= plt.subplots(nrows=num_rows, ncols=2, figsize=(16, 6*num_rows))
            axes = axes.flat
            fig.suptitle('Distribución variables categóricas',fontsize = 18)
            print(object_columns)
            for i, colum in enumerate(object_columns):
                print(i,colum)
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
            print(f"[INFO] {self.img_path}{file_name} creado correctamente")  
        else:
            print(f"[INFO] La imagen {file_name} ya existe")

    def createAndSaveViolinPlot(self, df: pd.DataFrame, file_name : str, target_col: str= 'Price', columns:list = None):
        """Crea graficos violin de todos las dimensiones de df con tipo ['object', 'uint8'] y las guarda en la ruta img_path, si se le pasa una lista con 
            dimensiones del df unicamente graficará dicha lista si se le pasa columns por parámetro graficará dichas columnas en base al target_col
        :param df: Dataframe
        :param file_name: Nombre y extension del archivo
        :param target_col: nombre de la columna etiqueta, por defecto el Price
        :param columns: por defecto None, lo cual implica que grafica todos las dimensiones del df, 
                        se puede pasar una lista con las columnas que se desea graficar """ 
        print('[INFO] Creando imagen gráfica...')
        if not os.path.exists(f'{self.img_path}{file_name}'):
            
            if columns:
                object_columns = df[columns].select_dtypes(include=['object']).columns
                uint8_columns = df[columns].select_dtypes(include=['uint8']).columns
            else:
                object_columns = df.select_dtypes(include=['object']).columns
                uint8_columns = df.select_dtypes(include=['uint8']).columns
            object_axes = list(zip(object_columns, range(0,len(object_columns))))
            uint8_axes = list(zip(uint8_columns, range(len(object_columns),len(object_columns)+len(uint8_columns))))
            size= math.ceil((len(object_columns)+len(uint8_columns))/2)
            fig, axes= plt.subplots(nrows=size, ncols=2, figsize=(16, 6*size))
            axes = axes.flat
            fig.suptitle('Distribución del precio por variable categórica',va='top', y= 1,fontsize = 18)
            print(len(axes))
            print(len(object_axes))
            print(len(uint8_axes))
            for colum, i in object_axes:
                print(i)
                print(colum)
                sns.violinplot(
                    x     = colum,
                    y     = target_col,
                    data  = df,
                    color = "white",
                    ax    = axes[i]
                )
                axes[i].set_title(f"precio vs {colum}", fontsize = 7, fontweight = "bold")
                axes[i].yaxis.set_major_formatter(ticker.EngFormatter())
                axes[i].tick_params(labelsize = 6)
                axes[i].set_xlabel("")
                axes[i].set_ylabel("")

            for colum, i in uint8_axes:
                print(i)
                print(colum)
                sns.violinplot(
                    x     = target_col,
                    y     = colum,
                    data  = df,
                    color = "white",
                    ax    = axes[i]
                )
                axes[i].set_title(f"precio vs {colum}", fontsize = 7, fontweight = "bold")
                axes[i].yaxis.set_major_formatter(ticker.EngFormatter())
                axes[i].tick_params(labelsize = 6)
                axes[i].set_xlabel("")
                axes[i].set_ylabel("")
            fig.tight_layout()
            fig.savefig(f"{self.img_path}{file_name}", bbox_inches="tight")
            print(f"{self.img_path}{file_name} creado correctamente")

            
            
        else:
            print(f"[INFO] La imagen {file_name} ya existe")  

    def createAndSaveCorrelationMatrix(self, df: pd.DataFrame, file_name : str, target_col: str= 'Price'):
        """Crea la matriz de correlación de las dimensiones numéricas del df
            :param df : Dataframe
            :param file_name: Nombre y extension del archivo
           """
        print('[INFO] Creando imagen gráfica...')
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
            print(f"[INFO] {self.img_path}{file_name} creado correctamente") 
        else:
            print(f"[INFO] La imagen {file_name} ya existe")

    def createAndSaveplotGridValues(self, vector, scores,file_name: str, metric: str, n_folds: int, x_label: str):
        """"""
        print('[INFO] Creando imagen gráfica...')
        if not os.path.exists(f'{self.img_path}{file_name}'):
            fig, ax = plt.subplots()
            ax.semilogx(vector,scores,'-o')
            plt.xlabel(x_label,fontsize=16)
            plt.ylabel(f'{n_folds}-Fold MSE')
            plt.savefig(f"{self.img_path}{file_name}")
            print(f"{self.img_path}{file_name} creado correctamente")
        else:
            print(f"[INFO] La imagen {file_name} ya existe")

    def createAndSaveModelsCompare(self, models: pd.DataFrame, file_name: str) -> None:
        """"""
        print('[INFO] Creando imagen gráfica...')
        if not os.path.exists(f'{self.img_path}{file_name}'):
            metrics = [metric for metric in models.columns if metric != 'model']
            fig, axes= plt.subplots(nrows=3, ncols=2, figsize=(16, 12))
            axes = axes.flat
            fig.suptitle('Comparación de modelos',va='top', y= 1,fontsize = 18)
            for i, metric in enumerate(metrics):
                 sns.barplot(x=models[metric], y=models['model'], palette='Reds', ax=axes[i])
                 axes[i].tick_params(labelsize = 10)
                 axes[i].set_xlabel(f'Test {metric}', fontsize=12)
                 axes[i].set_ylabel('Modelos', fontsize=12)
                 axes[i].set_title(f'Comparación de {metric} de test modelos', fontsize=14)
            fig.tight_layout()
            plt.savefig(f"{self.img_path}{file_name}")
            print(f"{self.img_path}{file_name} creado correctamente")         

        else:
            print(f"[INFO] La imagen {file_name} ya existe")

    @staticmethod
    def printText(text_to_print: str):
        print(f'\n#############################################################################################################################\n'
              f'----    {text_to_print}\n'
              f'###############################################################################################################################\n')



