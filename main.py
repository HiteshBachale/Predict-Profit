"""
In the Main file we are going to load data and call respected functions for model
development
"""
import numpy as np
import pandas as pd
import sklearn
import sys
import matplotlib.pyplot as plt
import logging
import pickle
from MLR_Log import setup_logging
logger = setup_logging('main')
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
class MLR_INFO:
    try:
        def __init__(self,path):
            self.df = pd.read_csv(path)
            logger.info(f"Data Loaded Successfull : {self.df.shape}")
            logger.info(f"Sample Rows And Columns Data : {self.df.sample(7)}")
            # Checking categorical, unique values in State column
            logger.info(f'Categorical Unique Values: {self.df['State'].unique()}')
            # Checking categorical, unique values counts in State column
            logger.info(f'{self.df['State'].value_counts()}')
            # Converting categorical, unique values counts data into numerical values in State column
            self.df['State'] = self.df['State'].map({'New York': 0, 'California': 1, 'Florida': 2}).astype(int)
            # Checking the head first 5 records after converting categorical, unique values counts data into numerical values in State column
            logger.info(f'{self.df.sample(7)}')
            #self.df = self.df.drop(['Id'],axis=1)
            self.X = self.df.iloc[: , : -1] # independent data
            self.y = self.df.iloc[: , -1] # dependent data
            #logger.info(f'{self.X}')
            #logger.info(f'{self.y}')
            # checking if the data is clean or not:
            logger.info(f"Missing Value in the data : {self.df.isnull().sum()}")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size=0.3,random_state=42)
            logger.info(f'length of X_train and y_train : {len(self.X_train), len(self.y_train)}')
            logger.info(f'length of X_test and y_test : {len(self.X_test), len(self.y_test)}')

    except Exception as e:
        er_ty,er_msg,er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def algo_p1(self):
        try:
            # Give the data to Linear Regression Algorithm
            logger.info("Linear Regression Algorithm")
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            # Ensure X is 2D
            self.X_train = self.X_train.to_frame() if isinstance(self.X_train, pd.Series) else self.X_train
            self.X_test = self.X_test.to_frame() if isinstance(self.X_test, pd.Series) else self.X_test
            self.mli_reg = LinearRegression()
            self.mli_reg.fit(self.X_train, self.y_train)
            y_test_pred = self.mli_reg.predict(self.X_test)
            # Multiple Linear Regression -> y = mx + c, Finding m value
            logger.info(f'Finding m value : {self.mli_reg.coef_}')
            # Multiple Linear Regression -> y = mx + c, Finding c value
            logger.info(f'Finding c value : {self.mli_reg.intercept_}')
            # Regression metrics
            logger.info(f"Mean Squared Error : {mean_squared_error(self.y_test, y_test_pred)}")
            logger.info(f"Mean Absolute Error: {mean_absolute_error(self.y_test, y_test_pred)}")
            logger.info(f"R2 Score           : {r2_score(self.y_test, y_test_pred)*100}")

            # Give the data to Train Performance
            # Creating dataframe for traning data -> X_train as X_Train_Values and y_train as Y_Train_Values
            training_data = pd.DataFrame()
            training_data = self.X_train.copy()  # Shallow Copy
            training_data['Actual_Profit_Values'] = self.y_train
            # Calling dataframe for traning data -> X_train with Actual_Profit_Values
            logger.info(f'{training_data}')

            # Example for calculating model with the above training_data with y = mx + c
            p = 8.05287266e-01 * 134615.46 - 9.09759187e-02 * 147198.87 + 2.76195629e-02 * 127716.82 + 8.47949958e+02 * 1 + 56350.35199301147
            logger.info(f'Calculating model with the above training_data with y = mx + c : {p}')

            # Train Performance
            # Storing X_train Prediction data using Multiple Linear Regression As 'mli_reg' In Y_train_pred
            Y_train_pred = self.mli_reg.predict(self.X_train)
            # Dataframe for traning data and predicted data comparison
            training_data['Values_From_Model'] = Y_train_pred
            # Checking dataframe for traning data and predicted data comparison
            logger.info(training_data)
            # Regression metrics
            logger.info(f"Training Loos : {mean_squared_error(self.y_train,Y_train_pred)}")
            #logger.info(f"Mean Absolute Error: {mean_absolute_error(self.y_train,Y_Train_Predictions)}")
            logger.info(f"Training Accuracy :  {r2_score(self.y_train,Y_train_pred)*100}")

            # Test Performance
            # Storing X_test Prediction data using Multiple Linear Regression As 'mli_reg' In Y_test_pred
            self.Y_test_pred = self.mli_reg.predict(self.X_test)
            # Regression metrics
            logger.info(f"Testing Loos : {mean_squared_error(self.y_test, self.Y_test_pred)}")
            # logger.info(f"Mean Absolute Error: {mean_absolute_error(self.y_train,Y_Train_Predictions)}")
            logger.info(f"Testing Accuracy :  {r2_score(self.y_test, self.Y_test_pred) * 100}")

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    '''def mli_pred(self):
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

            # Checking X_train columns for calculations applying in y = mx + c
            logger.info(f'***** Checking X_train columns for calculations applying in y = mx + c *****')
            logger.info(self.X_train.columns)
            # Random example for Algorithm -> y = m1x1 + m2x2 + m3x3 + m4x4 + c , i.e y = mx + c
            outcome = 8.05287266e-01 * 5000 - 9.09759187e-02 * 12000 + 2.76195629e-02 * 18000 + 8.47949958e+02 * 2 + 56350.35199301147
            logger.info(f'{outcome}')

            #logger.info(f'********** Checking Predictions For New Data Points Externally **********')
            # Importing warnings library to ignore or filter warnings
            import warnings
            warnings.filterwarnings('ignore')
            # Predicting values based on user inputs
            out = self.mli_reg.predict([[5000, 12000, 18000, 2]])
            logger.info(f'Prediction Value was : {out[0]} : with {r2_score(self.y_test, self.Y_test_pred) * 100} % Accuracy')

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')'''

    def mli_pkl(self):
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

            # Writing the file
            # w -> write format, wb -> write with binary format
            with open('MLR_Model.pkl', 'wb') as f:
                pickle.dump(self.mli_reg, f)

            # Reading the file
            # Checking the file
            with open('MLR_Model.pkl', 'rb') as f:
                m = pickle.load(f)

            # Predicting values based on user inputs
            logger.info(f'***** Predicting values based on user inputs pickle files *****')
            out = m.predict([[5000, 12000, 18000, 2]])
            #out = m.predict([[6000, 14000, 20000, 4]])
            logger.info(f'Prediction Value was : {out[0]} : with {r2_score(self.y_test, self.Y_test_pred) * 100} % Accuracy')

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')


if __name__ == "__main__":
    try:
        path = 'E:\\Aspire Tech Academy Bangalore\\Data Science Tools\\Machine Learning\\Machine Learning Projects\\Multiple Linear Regression\\50_Startups.csv'
        obj = MLR_INFO(path)
        obj.algo_p1()
        #obj.mli_pred()
        obj.mli_pkl()
    except Exception as e:
        er_ty,er_msg,er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')













