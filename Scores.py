import joblib


def Scores(classification_scores,regression_scores,best_classification_model,best_regression_model):
    # Print classification model scores
    print("Classification Model Scores:")
    for name, score in classification_scores:
        print(f"{name}: {score}")
        
    # Print regression model RMSE
    print("\nRegression Model RMSE:")
    for name, score in regression_scores:
        print(f"{name}: {score}")
        
    # Print best classification model
    print("\nBest Classification Model:")
    print(f"Model: {best_classification_model[0]}")
    print(f"Accuracy Score: {best_classification_model[1]}")
    
    # Print best regression model
    print("\nBest Regression Model:")
    print(f"Model: {best_regression_model[0]}")
    print(f"RMSE Score: {best_regression_model[1]}")
    
    
    # Save the best classification model
    joblib.dump(best_classification_model, 'saved_models/best_classification_model.pkl') 
    print("\nClassification model saved....!")

    # Save the best regression model
    joblib.dump(best_regression_model, 'saved_models/best_regression_model.pkl')
    print("\nRegression model saved....!")