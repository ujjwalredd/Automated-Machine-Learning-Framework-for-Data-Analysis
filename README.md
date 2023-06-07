# Automated Machine Learning Framework for Data Analysis.
A machine learning framework for automated data analysis tasks is provided by this code. For feature selection, model training, and model assessment on datasets containing both numerical and categorical characteristics, it makes use of a variety of machine learning methods and techniques.

# Features
* Automated feature selection: The framework has techniques for working with datasets that have both category and numerical characteristics. To choose the most pertinent characteristics for analysis, it employs the proper feature selection algorithms.
* For both classification and regression tasks, the code uses a number of machine learning methods, including decision trees, support vector machines, logistic regression, random forests, and multi-layer perceptrons.
* Optimisation of hyperparameters: The framework uses grid search to automatically improve the hyperparameters of the chosen machine learning models, enhancing their performance on the provided dataset.
* Evaluation of cross-validation: To assess the efficacy of each model, the code carries out cross-validation. When doing classification or regression tasks, it employs accuracy ratings and mean squared error (MSE).

# Usage
* Ensure you have Python 3.x installed on your system.
* Install the required dependencies by running the following command: `pip install -r requirements.txt`
* Prepare your dataset in a compatible format. The dataset should be a CSV file with the target variable in a separate column.
* Modify the code to load your dataset and specify the relevant columns for features and the target variable.
* Run the code using a Python interpreter (`main.py`).
* The code will automatically perform feature selection, train multiple machine learning models, optimize their hyperparameters, and evaluate their performance.
* The best classification model and the best regression model will be displayed, along with their respective accuracy scores or MSE scores.

# Dependencies
The code requires the following dependencies to be installed:
* pandas
* numpy
* scikit-learn
You can install the dependencies using the following command: "pip install -r requirements.txt"

# License
This code is released under the [MIT License](LICENSE).

Feel free to modify and use the code according to your needs. Contributions are welcome!

# Contact
If you have any questions or suggestions regarding the code, please feel free to contact Ujjwal Reddy K S at ujjwalreddyks@gmail.com
