from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, f_regression
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from yellowbrick.regressor import PredictionError
from yellowbrick.regressor import ResidualsPlot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def rmse(y_test, y_pred):
    """
    Definition iof the RMSE (Root Mean Square Error)
    """
    return np.sqrt(mean_squared_error(y_test, y_pred))


def linear_regression(X_train, y_train, X_test, y_test, plot):
    """
    Performing a simple linear regression
    """
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_pred = linreg.predict(X_test)
    # Extract the feature importance
    coef = pd.Series(linreg.coef_, index=X_train.columns)
    imp_coef = coef.sort_values()
    # Plot the feature importance
    if plot:
        plt.rcParams["figure.figsize"] = (8.0, 10.0)
        imp_coef.plot(kind="barh")
        plt.title("Feature importance using Linear Model")
    # Return metrics
    return {
        "name": "Linear Regression (Baseline)",
        "R squared": linreg.score(X_test, y_test),
        "R squared training": linreg.score(X_train, y_train),
        "RMSE": rmse(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
    }


def lasso_regression(X_train, y_train, X_test, y_test, plot):
    """
    Perfomring a lasso regression with built in CV and plotting the feature importance
    """
    # Fit the ridge regression
    reg = LassoCV()    
    reg.fit(X_train, y_train)
    print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" % reg.score(X_train, y_train))
    coef = pd.Series(reg.coef_, index=X_train.columns)
    print(
        "Lasso picked "
        + str(sum(coef != 0))
        + " variables and eliminated the other "
        + str(sum(coef == 0))
        + " variables"
    )
    # Extract the feature importance
    imp_coef = coef.sort_values()
    # Plot the feature importance
    if plot:
        plt.rcParams["figure.figsize"] = (8.0, 10.0)
        imp_coef.plot(kind="barh")
        plt.title("Feature importance using Lasso Model")
        plt.show()

        # Plotting the prediction error
        visualizer = PredictionError(reg, size=(1080, 720))
        visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
        visualizer.score(X_test, y_test)  # Evaluate the model on the test data
        visualizer.show()                 # Finalize and render the figure
        # Visualizing the regression
        visualizer = ResidualsPlot(reg, size=(1080, 720))
        visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
        visualizer.score(X_test, y_test)  # Evaluate the model on the test data
        visualizer.show()                 # Finalize and render the figure
    # Using the test data to calculate a score
    y_pred = reg.predict(X_test)
    # Return metrics
    return {
        "name": "Lasso Regression",
        "R squared": reg.score(X_test, y_test),
        "RMSE": rmse(y_test, y_pred),
        "R squared training": reg.score(X_train, y_train),
        "MAE": mean_absolute_error(y_test, y_pred),
    }


def ridge_regression(X_train, y_train, X_test, y_test, plot):
    """
    Perfomring a ridge regression with built in CV and plotting the feature importance
    """
    # Fit the ridge regression
    reg = RidgeCV()
    reg.fit(X_train, y_train)
    print("Best alpha using built-in RidgeCV: %f" % reg.alpha_)
    print("Best score using built-in RidgeCV: %f" % reg.score(X_train, y_train))
    coef = pd.Series(reg.coef_, index=X_train.columns)
    print(
        "Ridge picked "
        + str(sum(coef != 0))
        + " variables and eliminated the other "
        + str(sum(coef == 0))
        + " variables"
    )
    # Extract the feature importance
    imp_coef = coef.sort_values()
    # Plot the feature importance
    if plot:
        plt.rcParams["figure.figsize"] = (8.0, 10.0)
        imp_coef.plot(kind="barh")
        plt.title("Feature importance using Ridge Model")
        plt.show()
        # Visualizing the regression
        visualizer = ResidualsPlot(reg, size=(1080, 720))
        visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
        visualizer.score(X_test, y_test)  # Evaluate the model on the test data
        visualizer.show()                 # Finalize and render the figure
    # Using the test data to calculate a score
    y_pred = reg.predict(X_test)
    # Return metrics
    return {
        "name": "Ridge Regression",
        "R squared": reg.score(X_test, y_test),
        "R squared training": reg.score(X_train, y_train),
        "RMSE": rmse(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
    }


def random_forest_regression(X_train, y_train, X_test, y_test, plot):
    """
    Random Forest Regression using grid search for the parameter tuning 
    and plotting the feature importances
    """
    # Random forest regressor
    rf = RandomForestRegressor(random_state=0)
    # Grid search for parameter tuning
    params = {
        "n_estimators": [10, 20, 30],
        "max_features": ["auto", "log2", "sqrt"],
        "bootstrap": [True, False],
    }
    reg = GridSearchCV(rf, params, cv=5)
    reg.fit(X_train, y_train)
    estimator = reg.best_estimator_
    # Using the test data to calculate a score
    y_pred = estimator.predict(X_test)
    print("Score on test data: ", estimator.score(X_test, y_test))
    print("Root Mean Square Error: ", rmse(y_test, y_pred))
    # Plotting the feature importances
    if plot:
        feat_importances = pd.Series(
            estimator.feature_importances_, index=X_train.columns
        ).sort_values(ascending=True)
        feat_importances.plot(kind="barh")
    return {
        "name": "Random Forest Regression",
        "R squared": estimator.score(X_test, y_test),
        "R squared training": estimator.score(X_train, y_train),
        "RMSE": rmse(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
    }
