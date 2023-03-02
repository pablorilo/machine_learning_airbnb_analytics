import math
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import  LinearRegression,Ridge,Lasso 
from sklearn.ensemble import RandomForestRegressor

from graphics import Graphics

class Modeling(Graphics):
    def __init__(self, data_dict: dict, show_more_info: bool = True):
        #Instanciamos la superclase de Graphics para tener acceso desde la clase 
        super().__init__()
        self.show_more_info = show_more_info

        #Almacenamos en variables de la clase los datos obtenidos en preprocesing
        #conjunto de train
        self.x_train_norm = data_dict['x_train_norm']
        self.y_train_norm = data_dict['y_train_norm']
        #conjunto de test
        self.x_test_norm = data_dict['x_test_norm']
        self.y_test_norm = data_dict['y_test_norm']
        
    def run(self, method = str):
        """Dependiendo del método que se pase por parámetro se ejecutara el entrenamiento del mismo
        :param method: algoritmo con el que queremos entrenar el modelo"""
        if method == "RIDGE":
            self.ridge(self.x_train_norm)

        elif method == "LINEAR":
            self.linearRegresion(self.x_train_norm)
        
        elif method == "LASSO":
            self.lasso(self.x_train_norm)
        
        elif method == "FOREST":
            self.randomForest(self.x_train_norm)
        

    def linearRegresion(self, x_train_norm):
        self.printText('Modeling 1 : Realizamos el entrenamiento de una Regresión lineal')
        df_score = pd.DataFrame(columns=['model','mse_train','mae_train','mse_test','mae_test','r2_train','r2_test'])
        curr_cols =  list(self.x_train_norm.columns)
        excl_cols = curr_cols
        
        iter_security_lock = 0
        while len(excl_cols) != 0:
            print(f'\n>>>>>>>>>>>>>>>>   Current iteration: {iter_security_lock} starting from 0   <<<<<<<<<<<<<<<<')
            n_predictors= len(curr_cols)
            curr_cols, excl_cols, dict_score = self.linearInfo(self.x_train_norm[curr_cols], self.y_train_norm, self.x_test_norm[curr_cols],
                                                  self.y_test_norm)
            values = list(dict_score.values())
            values.insert(0,f'model_lr_{n_predictors}_features') 
            new_row = pd.DataFrame([values], columns=df_score.columns)
            df_score = pd.concat([df_score, new_row], ignore_index=True)
            print('\n')
            print(df_score)
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if iter_security_lock == 10:
                break

            iter_security_lock += 1
        print('\n Generamos gráficos comparativos de diferentes metricas para cada modelo probado\n')
        self.createAndSaveModelsCompare(models = df_score, file_name ='regresionmodelsCompare.png')
        
    def linearInfo(self, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array): 

        print(f'\n----------   Tenemos {x_train.shape[1]} columnas')
        linear = LinearRegression()
        linear.fit(x_train, y_train)

        y_pred_train = linear.predict(x_train)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)

        y_pred_test = linear.predict(x_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)

        r2_train = linear.score(x_train, y_train)
        r2_test = linear.score(x_test, y_test)

        print(f'El MSE de train es : {mse_train}')
        print(f'El MSE de test es : {mse_test}')
        print(f'El MAE de train es : {mae_train}')
        print(f'El MAE de test es : {mae_test}')
        print(f'El RMSE de train es : {math.sqrt(mse_train)}')
        print(f'El RMSE de test es : {math.sqrt(mse_test)}')
        print(f'El r2 de train es : {r2_train}')
        print(f'El r2 de test es : {r2_test}')

        dict_score = {
            'mse_train':mse_train,
            'mae_train':mae_train,
            'mse_test':mse_test,
            'mae_test':mae_test,
            'r2_train':r2_train,
            'r2_test':r2_test
        }

        sfm = SelectFromModel(linear)
        sfm.fit(x_train, y_train)
        # Pintamos las mejores caracteristicas
        selected_features = list(x_train.columns[sfm.get_support()])
        excluded_features = [z for z in list(x_train.columns) if z not in selected_features]
        print(f'\nExcluded features: {len(excluded_features)}/{len(x_train.columns)}: {excluded_features}')
        print(f'Current Features: {selected_features}')

        return selected_features, excluded_features, dict_score

    #Regularizació  de Ridge
    def ridge(self, XtrainScaled):
        self.printText("Modeling 2: Aplicamos RIDGE e iteramos hasta encontrar el mejor numero de dimensiones para el modelo")
        ridge_best_alpha = self.ridgeGridSearchCV(XtrainScaled, self.y_train_norm)
        df_score = pd.DataFrame(columns=['model','mse_train','mae_train','mse_test','mae_test','r2_train','r2_test'])
        curr_cols =  list(self.x_train_norm.columns)
        excl_cols = curr_cols

        iter_security_lock = 0

        while len(excl_cols) != 0:
            print(f'\n>>>>>>>>>>>>>>>>   Current iteration: {iter_security_lock} starting from 0   <<<<<<<<<<<<<<<<')
            n_predictors = len(curr_cols)
            curr_cols, excl_cols, dict_score= self.ridgeInfo(self.x_train_norm[curr_cols], self.y_train_norm, self.x_test_norm[curr_cols],
                                                  self.y_test_norm, ridge_best_alpha)
            values = list(dict_score.values())
            values.insert(0,f'model_lr_{n_predictors}_features') 
            new_row = pd.DataFrame([values], columns=df_score.columns)
            df_score = pd.concat([df_score, new_row], ignore_index=True)
            print('\n')
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if iter_security_lock == 10:
                break

            iter_security_lock += 1
        print('\n Generamos gráficos comparativos de diferentes metricas para cada modelo probado\n')
        self.createAndSaveModelsCompare(models = df_score, file_name ='ridgemodelsCompare.png')

    def ridgeGridSearchCV(self, x_train: np.array, y_train: np.array, logspace=np.logspace(-5, 1.8, 25),
                          cv_number: int = 5):
        print('[INFO] Realizando cross validation ...')
        alpha_vector = logspace
        param_grid = {'alpha': alpha_vector}
        grid = GridSearchCV(Ridge(), scoring='neg_mean_squared_error', param_grid=param_grid, cv=cv_number)
        grid.fit(x_train, y_train)
        print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
        print("best parameters: {}".format(grid.best_params_))

        scores = -1 * np.array(grid.cv_results_['mean_test_score'])
        print('Generamos gráfica de comparación de grid\n')
        self.createAndSaveplotGridValues(vector= alpha_vector, scores= scores, file_name= f"AlphaRidge{x_train.shape[1]}.png", metric= 'Alpha', n_folds= 5, x_label= 'Alpha')

        # Devolvemos el mejor aplpha
        return grid.best_params_['alpha']

    
    def ridgeInfo(self,x_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array, alpha: float):
        print(f'\n----------   Tenemos {x_train.shape[1]} columnas')
        
        ridge = Ridge(alpha=alpha)
        ridge.fit(x_train, y_train)

        y_pred_train = ridge.predict(x_train)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)

        y_pred_test = ridge.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)

        r2_train = ridge.score(x_train, y_train)
        r2_test = ridge.score(X_test, y_test)

        print(f'El MSE de train es : {mse_train}')
        print(f'El MSE de test es : {mse_test}')
        print(f'El MAE de train es : {mae_train}')
        print(f'El MAE de test es : {mae_test}')
        print(f'El RMSE de train es : {math.sqrt(mse_train)}')
        print(f'El RMSE de test es : {math.sqrt(mse_test)}')
        print(f'El r2 de train es : {r2_train}')
        print(f'El r2 de test es : {r2_test}')

        dict_score = {
            'mse_train':mse_train,
            'mae_train':mae_train,
            'mse_test':mse_test,
            'mae_test':mae_test,
            'r2_train':r2_train,
            'r2_test':r2_test
        }

        sfm = SelectFromModel(ridge)
        sfm.fit(x_train, y_train)
        
        # Pintamos las mejores caracteristicas
        selected_features = list(x_train.columns[sfm.get_support()])
        excluded_features = [z for z in list(x_train.columns) if z not in selected_features]
        print(f'\nExcluded features: {len(excluded_features)}/{len(x_train.columns)}: {excluded_features}')
        print(f'Current Features: {selected_features}')
        return selected_features, excluded_features, dict_score

    # -- LASSO
    def lasso(self, XtrainScaled):
        self.printText("Modeling 3: Aplicamos LASSO e iteramos hasta encontrar el mejor numero de predictores")
        lasso_best_alpha = self.lassoGridSearchCV(XtrainScaled, self.y_train_norm)
        df_score = pd.DataFrame(columns=['model','mse_train','mae_train','mse_test','mae_test','r2_train','r2_test'])
        curr_cols = list(self.x_train_norm.columns)
        excl_cols = curr_cols

        iter_security_lock = 0

        while len(excl_cols) != 0:
            n_predictors= len(curr_cols)
            print(f'\n>>>>>>>>>>>>>>>>   Current iteration: {iter_security_lock} starting from 0   <<<<<<<<<<<<<<<<')
            curr_cols, excl_cols, dict_score = self.lassoInfo(self.x_train_norm[curr_cols], self.y_train_norm, self.x_test_norm[curr_cols],
                                                  self.y_test_norm, lasso_best_alpha)
            values = list(dict_score.values())
            values.insert(0,f'model_lr_{n_predictors}_features') 
            new_row = pd.DataFrame([values], columns=df_score.columns)
            df_score = pd.concat([df_score, new_row], ignore_index=True)
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if iter_security_lock == 10:
                break

            iter_security_lock += 1
        print('\n Generamos gráficos comparativos de diferentes metricas para cada modelo probado\n')
        self.createAndSaveModelsCompare(models = df_score, file_name ='lassomodelsCompare.png')

    def lassoGridSearchCV(self, X_train: np.array, y_train: np.array, logspace=np.logspace(-5, 1.8, 25),
                          cv_number: int = 5):
        print('[INFO] Realizando cross validation ...')
        alpha_vector = logspace
        param_grid = {'alpha': alpha_vector}
        grid = GridSearchCV(Lasso(), scoring='neg_mean_squared_error', param_grid=param_grid, cv=cv_number)
        grid.fit(X_train, y_train)
        print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
        print("best parameters: {}".format(grid.best_params_))

        scores = -1 * np.array(grid.cv_results_['mean_test_score'])
        print('Generamos gráfica de comparación de grid\n')
        self.createAndSaveplotGridValues(vector= alpha_vector, scores= scores, file_name= f"AlphaLasso{X_train.shape[1]}.png", metric= 'Alpha', n_folds= 5, x_label= 'Alpha')

        # Devolvemos el mejor aplpha
        return grid.best_params_['alpha']

    
    def lassoInfo(self,X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array, alpha: float):
        print(f'\n----------   Tenemos {X_train.shape[1]} columnas en training')
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train, y_train)

        y_pred_train = lasso.predict(X_train)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)

        y_pred_test = lasso.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)

        r2_train = lasso.score(X_train, y_train)
        r2_test = lasso.score(X_test, y_test)

        print(f'El MSE de train es : {mse_train}')
        print(f'El MSE de test es : {mse_test}')
        print(f'El MAE de train es : {mae_train}')
        print(f'El MAE de test es : {mae_test}')
        print(f'El RMSE de train es : {math.sqrt(mse_train)}')
        print(f'El RMSE de test es : {math.sqrt(mse_test)}')
        print(f'El r2 de train es : {r2_train}')
        print(f'El r2 de test es : {r2_test}')

        dict_score = {
            'mse_train':mse_train,
            'mae_train':mae_train,
            'mse_test':mse_test,
            'mae_test':mae_test,
            'r2_train':r2_train,
            'r2_test':r2_test
        }

        sfm = SelectFromModel(lasso)
        sfm.fit(X_train, y_train)

        # Pintamos las mejores caracteristicas
        selected_features = list(X_train.columns[sfm.get_support()])
        excluded_features = [z for z in list(X_train.columns) if z not in selected_features]
        print(f'\nExcluded features: {len(excluded_features)}/{len(X_train.columns)}: {excluded_features}')
        print(f'Current Features: {selected_features}')

        return selected_features, excluded_features, dict_score

    # -- RANDOM FOREST
    def randomForest(self, XtrainScaled):
        self.printText("Modeling 4: Aplicamos RandomForest e iteramos hasta encontrar el mejor numero de predictores")
        rf_best_alpha = self.randomForestGridSearchCV(XtrainScaled, self.y_train_norm)
        df_score = pd.DataFrame(columns=['model','mse_train','mae_train','mse_test','mae_test','r2_train','r2_test'])
        curr_cols = list(self.x_train_norm.columns)
        excl_cols = curr_cols

        iter_security_lock = 0

        while len(excl_cols) != 0:
            n_predictors= len(curr_cols)
            print(f'\n>>>>>>>>>>>>>>>>   Current iteration: {iter_security_lock} starting from 0   <<<<<<<<<<<<<<<<')
            curr_cols, excl_cols, dict_score = self.randomForestInfo(self.x_train_norm[curr_cols], self.y_train_norm, self.x_test_norm[curr_cols],
                                                         self.y_test_norm, rf_best_alpha)
            values = list(dict_score.values())
            values.insert(0,f'model_lr_{n_predictors}_features') 
            new_row = pd.DataFrame([values], columns=df_score.columns)
            df_score = pd.concat([df_score, new_row], ignore_index=True)
            print('\n')
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if iter_security_lock == 15:
                break

            iter_security_lock += 1
        print('\n Generamos gráficos comparativos de diferentes metricas para cada modelo probado\n')
        self.createAndSaveModelsCompare(models = df_score, file_name ='randommodelsCompare.png')

    def randomForestGridSearchCV(self, X_train: np.array, y_train: np.array, maxDepth=range(1, 18), cv_number: int = 5):
        print('[INFO] Realizando cross validation ...')
        param_grid = {'max_depth': maxDepth}
        grid = GridSearchCV(RandomForestRegressor(random_state=0, n_estimators=200, max_features='sqrt'),
                            param_grid=param_grid, cv=cv_number, verbose=2)
        grid.fit(X_train, y_train)
        print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
        print("best parameters: {}".format(grid.best_params_))

        scores = np.array(grid.cv_results_['mean_test_score'])
        print('Generamos gráfica de comparación de grid\n')
        self.createAndSaveplotGridValues(vector= maxDepth, scores= scores, file_name= f"randomforest{X_train.shape[1]}.png", metric= 'Max_depth', n_folds= 5, x_label= 'Alpha')

        # Devolvemos el mejor depth
        return grid.best_params_['max_depth']

    
    def randomForestInfo(self, X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array,
                         maxDepth: float):
        print(f'\n----------   Tenemos {X_train.shape[1]} columnas en training')
        print(f'----------   Tenemos {X_test.shape[1]} columnas en testing\n')
        rf = RandomForestRegressor(max_depth=maxDepth, n_estimators=200, max_features='sqrt')
        rf.fit(X_train, y_train)

        y_pred_train = rf.predict(X_train)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)

        y_pred_test = rf.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)

        r2_train = rf.score(X_train, y_train)
        r2_test = rf.score(X_test, y_test)

        print(f'El MSE de train es : {mse_train}')
        print(f'El MSE de test es : {mse_test}')
        print(f'El MAE de train es : {mae_train}')
        print(f'El MAE de test es : {mae_test}')
        print(f'El RMSE de train es : {math.sqrt(mse_train)}')
        print(f'El RMSE de test es : {math.sqrt(mse_test)}')
        print(f'El r2 de train es : {r2_train}')
        print(f'El r2 de test es : {r2_test}')

        dict_score = {
            'mse_train':mse_train,
            'mae_train':mae_train,
            'mse_test':mse_test,
            'mae_test':mae_test,
            'r2_train':r2_train,
            'r2_test':r2_test
        }

        sfm = SelectFromModel(rf)
        sfm.fit(X_train, y_train)

        # Pintamos las mejores caracteristicas
        selected_features = list(X_train.columns[sfm.get_support()])
        excluded_features = [z for z in list(X_train.columns) if z not in selected_features]
        print(f'\nExcluded features: {len(excluded_features)}/{len(X_train.columns)}: {excluded_features}')
        print(f'Current Features: {selected_features}')

        return selected_features, excluded_features, dict_score 
