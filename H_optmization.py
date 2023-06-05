from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import  MLPClassifier, MLPRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
import numpy as np

def H_optmization(x,y,featur):
    X = x[featur]
    Y = y      
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42) #train_test splitting 
    
    # Classification models
    classification_models = [
        ('Decision Tree', DecisionTreeClassifier(), {'max_depth': [3, 5, 10]}),
        ('Support Vector Machine', SVC(), {'C': [0.1, 1, 10]}),
        ('Logistic Regression', LogisticRegression(), {'C': [0.1, 1, 10]}),
        ('Random Forest', RandomForestClassifier(), {'n_estimators': [50, 100, 200]}),
        ('Multi-Layer Perceptron', MLPClassifier(), {'hidden_layer_sizes': [(50,), (100,), (200,)]})
    ]
    
   # Regression models
    regression_models = [
        ('Decision Tree', DecisionTreeRegressor(), {'max_depth': [3, 5, 10]}),
        ('Support Vector Machine', SVR(), {'C': [0.1, 1, 10]}),
        ('Linear Regression', LinearRegression(), {}),
        ('Random Forest', RandomForestRegressor(), {'n_estimators': [50, 100, 200]}),
        ('Multi-Layer Perceptron', MLPRegressor(), {'hidden_layer_sizes': [(50,), (100,), (200,)]})
    ]
    
    epochs = 5 #can be modifed as necessary 
    
    # Hyperparameter optimization for classification models
    classification_scores = []
    for name, model, param_grid in classification_models:
        try:
            grid_search = GridSearchCV(model, param_grid, cv=epochs, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            scores = cross_val_score(best_model, X_train, y_train, cv=epochs, scoring='accuracy')
            classification_scores.append((name, scores.mean()))
        except Exception as e:
            print()
            continue
            
    # Hyperparameter optimization for regression models
    regression_scores = []
    for name, model, param_grid in regression_models:
        try:
            grid_search = GridSearchCV(model, param_grid, cv=epochs, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            scores = np.sqrt(-cross_val_score(best_model, X_train, y_train, cv=epochs, scoring='neg_mean_squared_error'))
            regression_scores.append((name, np.mean(scores)))
        except Exception as e:
            print(f"Error occurred for {name}: {str(e)}")
            continue
        
    best_classification_model = max(classification_scores, key=lambda x: x[1]) # Find the best classification model
    best_regression_model = min(regression_scores, key=lambda x: x[1]) # Find the best regression model
    
    return(classification_scores, regression_scores,best_classification_model,best_regression_model)