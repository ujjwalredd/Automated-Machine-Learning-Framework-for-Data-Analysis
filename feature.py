import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def feature(x,y):

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) #train_test splitting
    
    # LASSO Regularization
    lasso = Lasso(alpha=0.1)  # Adjust the regularization strength (alpha) as needed
    lasso.fit(X_train, y_train)
    
    # Select features using LASSO regularization
    lasso_selector = SelectFromModel(lasso, prefit=True)
    selected_features_lasso = X_train.columns[lasso_selector.get_support()]
    
    # Recursive Feature Elimination (RFE) with Linear Regression
    linear_reg = LinearRegression()  # Use a regression model of your choice
    rfe_selector = RFE(linear_reg, n_features_to_select=10)  # Select the desired number of features
    rfe_selector.fit(X_train, y_train)
    
    
    selected_features_rfe = X_train.columns[rfe_selector.support_] # Select features using RFE with Linear Regression
    selected_features = list(set(selected_features_lasso) | set(selected_features_rfe)) # Compare the selected features
    
    # Train and evaluate models with selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    model_lasso = Lasso(alpha=0.1)  # Adjust the regularization strength (alpha) as needed
    model_lasso.fit(X_train_selected, y_train)
    predictions_lasso = model_lasso.predict(X_test_selected)
    lasso_rmse = np.sqrt(mean_squared_error(y_test, predictions_lasso))
    
    model_lasso = Lasso(alpha=0.1)  # Adjust the regularization strength (alpha) as needed
    model_lasso.fit(X_train_selected, y_train)
    predictions_lasso = model_lasso.predict(X_test_selected)
    lasso_rmse = np.sqrt(mean_squared_error(y_test, predictions_lasso))
    model_lasso = Lasso(alpha=0.1)  # Adjust the regularization strength (alpha) as needed
    model_lasso.fit(X_train_selected, y_train)
    predictions_lasso = model_lasso.predict(X_test_selected)
    lasso_rmse = np.sqrt(mean_squared_error(y_test, predictions_lasso))
    
    model_rfe = LinearRegression()
    model_rfe.fit(X_train_selected, y_train)
    predictions_rfe = model_rfe.predict(X_test_selected)
    rfe_rmse = np.sqrt(mean_squared_error(y_test, predictions_rfe))
    
    if lasso_rmse > rfe_rmse:
        featuresss = selected_features_rfe
    else:
        featuresss = selected_features_lasso
        
        
    return featuresss