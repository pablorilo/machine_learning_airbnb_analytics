import math
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
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
            self.ridgeApp(self.x_train_norm)

        elif method == "LINEAR":
            self.linearRegresion(self.x_train_norm)
        
        elif method == "LASSO":
            self.lassoApp(self.x_train_norm)
        

    def linearRegresion(self, x_train_norm):
        self.printText('Modeling 1 : Realizamos el entrenamiento de una Regresión lineal')
        
        curr_cols =  list(self.x_train_norm.columns)
        print(curr_cols)
        excl_cols = curr_cols

        iter_security_lock = 0
        while len(excl_cols) != 0:
            print(f'\n>>>>>>>>>>>>>>>>   Current iteration: {iter_security_lock} starting from 0   <<<<<<<<<<<<<<<<')
            curr_cols, excl_cols = self.linearInfo(self.x_train_norm[curr_cols], self.y_train_norm, self.x_test_norm[curr_cols],
                                                  self.y_test_norm)
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if iter_security_lock == 10:
                break

            iter_security_lock += 1
    def linearInfo(self, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array): 

        print(f'\n----------   Tenemos {x_train.shape[1]} columnas')
        linear = LinearRegression()
        linear.fit(x_train, y_train)

        y_pred_train = linear.predict(x_train)
        mse_train = mean_squared_error(y_train, y_pred_train)

        y_pred_test = linear.predict(x_test)
        mse_test = mean_squared_error(y_test, y_pred_test)

        r2_train = linear.score(x_train, y_train)
        r2_test = linear.score(x_test, y_test)

        print(f'El MSE de train es : {mse_train}')
        print(f'El MSE de test es : {mse_test}')
        print(f'El RMSE de train es : {math.sqrt(mse_train)}')
        print(f'El RMSE de test es : {math.sqrt(mse_test)}')
        print(f'El r2 de train es : {r2_train}')
        print(f'El r2 de test es : {r2_test}')

        sfm = SelectFromModel(linear)
        sfm.fit(x_train, y_train)
        # Pintamos las mejores caracteristicas
        selected_features = list(x_train.columns[sfm.get_support()])
        excluded_features = [z for z in list(x_train.columns) if z not in selected_features]
        print(f'\nExcluded features: {len(excluded_features)}/{len(x_train.columns)}: {excluded_features}')
        print(f'Current Features: {selected_features}')

        return selected_features, excluded_features

    #Regularizació  de Ridge
    def ridgeApp(self, XtrainScaled):
        self.printText("Modeling 2: Aplicamos RIDGE e iteramos hasta encontrar el mejor numero de dimensiones para el modelo")
        ridge_best_alpha = self.ridgeGridSearchCV(XtrainScaled, self.y_train_norm)

        curr_cols =  list(self.x_train_norm.columns)
        excl_cols = curr_cols

        iter_security_lock = 0

        while len(excl_cols) != 0:
            print(f'\n>>>>>>>>>>>>>>>>   Current iteration: {iter_security_lock} starting from 0   <<<<<<<<<<<<<<<<')
            curr_cols, excl_cols = self.ridgeInfo(self.x_train_norm[curr_cols], self.y_train_norm, self.x_test_norm[curr_cols],
                                                  self.y_test_norm, ridge_best_alpha)
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if iter_security_lock == 10:
                break

            iter_security_lock += 1

    def ridgeGridSearchCV(self, x_train: np.array, y_train: np.array, logspace=np.logspace(-5, 1.8, 25),
                          cv_number: int = 5):
        alpha_vector = logspace
        param_grid = {'alpha': alpha_vector}
        grid = GridSearchCV(Ridge(), scoring='neg_mean_squared_error', param_grid=param_grid, cv=cv_number)
        grid.fit(x_train, y_train)
        print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
        print("best parameters: {}".format(grid.best_params_))

        scores = -1 * np.array(grid.cv_results_['mean_test_score'])
        self.createAndSaveplotGridValues(vector= alpha_vector, scores= scores, file_name= f"AlphaRidge{x_train.shape[1]}.png", metric= 'Alpha', n_folds= 5, x_label= 'Alpha')

        # Devolvemos el mejor aplpha
        return grid.best_params_['alpha']

    
    def ridgeInfo(self,x_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array, alpha: float):
        print(f'\n----------   Tenemos {x_train.shape[1]} columnas')
        
        ridge = Ridge(alpha=alpha)
        ridge.fit(x_train, y_train)

        y_pred_train = ridge.predict(x_train)
        mse_train = mean_squared_error(y_train, y_pred_train)

        y_pred_test = ridge.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred_test)

        r2_train = ridge.score(x_train, y_train)
        r2_test = ridge.score(X_test, y_test)

        print(f'El MSE de train es : {mse_train}')
        print(f'El MSE de test es : {mse_test}')
        print(f'El RMSE de train es : {math.sqrt(mse_train)}')
        print(f'El RMSE de test es : {math.sqrt(mse_test)}')
        print(f'El r2 de train es : {r2_train}')
        print(f'El r2 de test es : {r2_test}')

        sfm = SelectFromModel(ridge)
        sfm.fit(x_train, y_train)
        
        # Pintamos las mejores caracteristicas
        selected_features = list(x_train.columns[sfm.get_support()])
        excluded_features = [z for z in list(x_train.columns) if z not in selected_features]
        print(f'\nExcluded features: {len(excluded_features)}/{len(x_train.columns)}: {excluded_features}')
        print(f'Current Features: {selected_features}')
        return selected_features, excluded_features

    # -- LASSO
    def lassoApp(self, XtrainScaled):
        self.printText("Modeling 3: Aplicamos LASSO e iteramos hasta encontrar el mejor numero de predictores")
        lasso_best_alpha = self.lassoGridSearchCV(XtrainScaled, self.y_train_norm)

        curr_cols = list(self.x_train_norm.columns)
        excl_cols = curr_cols

        iter_security_lock = 0

        while len(excl_cols) != 0:
            print(f'\n>>>>>>>>>>>>>>>>   Current iteration: {iter_security_lock} starting from 0   <<<<<<<<<<<<<<<<')
            curr_cols, excl_cols = self.lassoInfo(self.x_train_norm[curr_cols], self.y_train_norm, self.x_test_norm[curr_cols],
                                                  self.y_test_norm, lasso_best_alpha)
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if iter_security_lock == 10:
                break

            iter_security_lock += 1

    def lassoGridSearchCV(self, X_train: np.array, y_train: np.array, logspace=np.logspace(-5, 1.8, 25),
                          cv_number: int = 5):
        alpha_vector = logspace
        param_grid = {'alpha': alpha_vector}
        grid = GridSearchCV(Lasso(), scoring='neg_mean_squared_error', param_grid=param_grid, cv=cv_number)
        grid.fit(X_train, y_train)
        print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
        print("best parameters: {}".format(grid.best_params_))

        scores = -1 * np.array(grid.cv_results_['mean_test_score'])
        self.createAndSaveplotGridValues(vector= alpha_vector, scores= scores, file_name= f"AlphaLasso{X_train.shape[1]}.png", metric= 'Alpha', n_folds= 5, x_label= 'Alpha')

        # Devolvemos el mejor aplpha
        return grid.best_params_['alpha']

    
    def lassoInfo(self,X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array, alpha: float):
        print(f'\n----------   Tenemos {X_train.shape[1]} columnas en training')
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train, y_train)

        y_pred_train = lasso.predict(X_train)
        mse_train = mean_squared_error(y_train, y_pred_train)

        y_pred_test = lasso.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred_test)

        r2_train = lasso.score(X_train, y_train)
        r2_test = lasso.score(X_test, y_test)

        print(f'El MSE de train es : {mse_train}')
        print(f'El MSE de test es : {mse_test}')
        print(f'El RMSE de train es : {math.sqrt(mse_train)}')
        print(f'El RMSE de test es : {math.sqrt(mse_test)}')
        print(f'El r2 de train es : {r2_train}')
        print(f'El r2 de test es : {r2_test}')

        sfm = SelectFromModel(lasso)
        sfm.fit(X_train, y_train)

        # Pintamos las mejores caracteristicas
        selected_features = list(X_train.columns[sfm.get_support()])
        excluded_features = [z for z in list(X_train.columns) if z not in selected_features]
        print(f'\nExcluded features: {len(excluded_features)}/{len(X_train.columns)}: {excluded_features}')
        print(f'Current Features: {selected_features}')

        return selected_features, excluded_features

    # -- RANDOM FOREST
    def randomForestApp(self, XtrainScaled):
        self.introPrint("Modeling 4: Aplicamos RandomForest e iteramos hasta encontrar el mejor numero de predictores")
        rf_best_alpha = self.randomForestGridSearchCV(XtrainScaled, self.y_train)

        curr_cols = list(self.X_train.columns)
        excl_cols = curr_cols

        iter_security_lock = 0

        while len(excl_cols) != 0:
            print(f'\n>>>>>>>>>>>>>>>>   Current iteration: {iter_security_lock} starting from 0   <<<<<<<<<<<<<<<<')
            curr_cols, excl_cols = self.randomForestInfo(self.x_train_norm[curr_cols], self.y_train_norm, self.x_test_norm[curr_cols],
                                                         self.y_test_norm, rf_best_alpha)
            print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if iter_security_lock == 15:
                break

            iter_security_lock += 1

    def randomForestGridSearchCV(self, X_train: np.array, y_train: np.array, maxDepth=range(1, 18), cv_number: int = 5):

        param_grid = {'max_depth': maxDepth}
        grid = GridSearchCV(RandomForestRegressor(random_state=0, n_estimators=200, max_features='sqrt'),
                            param_grid=param_grid, cv=cv_number, verbose=2)
        grid.fit(X_train, y_train)
        print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
        print("best parameters: {}".format(grid.best_params_))

        scores = np.array(grid.cv_results_['mean_test_score'])
        self.plotAlphaValues(maxDepth, scores, f"AlphaRf{X_train.shape[1]}.png")

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

        y_pred_test = rf.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred_test)

        r2_train = rf.score(X_train, y_train)
        r2_test = rf.score(X_test, y_test)

        print(f'El MSE de train es : {mse_train}')
        print(f'El MSE de test es : {mse_test}')
        print(f'El RMSE de train es : {math.sqrt(mse_train)}')
        print(f'El RMSE de test es : {math.sqrt(mse_test)}')
        print(f'El r2 de train es : {r2_train}')
        print(f'El r2 de test es : {r2_test}')

        sfm = SelectFromModel(rf)
        sfm.fit(X_train, y_train)

        # Pintamos las mejores caracteristicas
        selected_features = list(X_train.columns[sfm.get_support()])
        excluded_features = [z for z in list(X_train.columns) if z not in selected_features]
        print(f'\nExcluded features: {len(excluded_features)}/{len(X_train.columns)}: {excluded_features}')
        print(f'Current Features: {selected_features}')

        return selected_features, excluded_features  
