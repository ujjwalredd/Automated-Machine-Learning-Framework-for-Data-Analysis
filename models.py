from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

def models(x,y,featur):
    X = x[featur]
    Y = y      
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42) #train_test splitting 
    # Classification models  can add more if needed
    classification_models = [
        ('Decision Tree', DecisionTreeClassifier()),
        ('Support Vector Machine', SVC()),
        ('Logistic Regression', LogisticRegression()),
        ('Random Forest', RandomForestClassifier()),
        ('Multi-Layer Perceptron', MLPClassifier())
    ]
    
    # Regression models can add more if needed
    regression_models = [
        ('Decision Tree', DecisionTreeRegressor()),
        ('Support Vector Machine', SVR()),
        ('Linear Regression', LinearRegression()),
        ('Random Forest', RandomForestRegressor()),
        ('Multi-Layer Perceptron', MLPRegressor())
    ]
    
    epochs = 5 #can be modifed as necessary 
    
    # Evaluate classification models
    classification_scores = []
    for name, model in classification_models:
        try:
            scores = cross_val_score(model, X_train, y_train, cv=epochs, scoring='accuracy')
            classification_scores.append((name, scores.mean()))
        except Exception as e:
            print()
            continue
        
    # Evaluate regression models
    regression_scores = []
    for name, model in regression_models:
        try:
            scores = -cross_val_score(model, X_train, y_train, cv=epochs, scoring='neg_mean_squared_error')
            regression_scores.append((name, np.sqrt(scores.mean())))
        except Exception as e:
            print()
            continue
        
    best_classification_model = max(classification_scores, key=lambda x: x[1]) # Find the best classification model
    best_regression_model = min(regression_scores, key=lambda x: x[1]) # Find the best regression model
    
    
    
    return(classification_scores, regression_scores,best_classification_model,best_regression_model)
    
