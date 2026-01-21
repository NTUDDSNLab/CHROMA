from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
import json
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.svm import SVR
import argparse

# m2cgen
import m2cgen as m2c

# import xgboost
import xgboost as xgb
from sklearn.linear_model import LinearRegression


def prepare_train_data(file_path):
    
    train_X = []
    train_y = []
    with open(file_path, 'r') as f:
        data = json.load(f)
        for dataset, value in data.items():
            #print("dataset: ", dataset)
            # print("data[dataset]: ", data[dataset]['0']['color'])
            base_theta_color = data[dataset]['0']['color']
            last_runtime = data[dataset]['0']['runtime_ms']
            #print("base_theta_color: ", base_theta_color)
            # Find the maximum theta with the same color as base_theta_color
            max_theta = 0
            for theta in data[dataset]:
                # Determine if theta is a number
                if theta == '0':
                    continue
                if theta == 'vertices' or theta == 'edges':
                    continue
                # Exclude None values
                if data[dataset][theta]['color'] == None:
                    continue
                
                cur_runtime = data[dataset][theta]['runtime_ms']
                cur_speedup = last_runtime / cur_runtime
                last_runtime = cur_runtime
                
                if data[dataset][theta]['color'] <= base_theta_color:
                    max_theta = theta
                    if cur_speedup <= 1.2:
                        break
                else:
                    break
            #print("Appending X: ", data[dataset]['vertices'], data[dataset]['edges'])
            #print("Appending y: ", max_theta)
        
            # Skip datasets without vertices/edges information
            if 'vertices' not in data[dataset] or 'edges' not in data[dataset]:
                print(f"Skipping {dataset}: missing vertices/edges information")
                continue
            
            train_X.append([data[dataset]['vertices'], data[dataset]['edges'], data[dataset]['vertices']/data[dataset]['edges']])
            train_y.append(int(max_theta))

            # print(dataset, " : ", max_theta)
    
    return np.array(train_X, dtype=int), np.array(train_y, dtype=int)



def violation_ratio(test, pred):
    violation_count = 0
    for i in range(len(test)):
        if test[i] < pred[i]:
            violation_count += 1
    return violation_count / len(test)



def random_forest_model(train_X, train_y, grid_search=False):
    # Split the data into training and testing sets
    print("----<<   SPLITTING DATA   >>----")
    train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)
    
    print("----<<   TRAINING MODEL   >>----")

    if grid_search:
        param_grid = {
        'n_estimators': [200, 500, 800],
        'max_depth': [100, 200, 300],
        'min_samples_split': [10, 20, 30],
        'min_samples_leaf': [5, 10, 20],
        'max_features': ['sqrt', 'log2', 0.5],
        'bootstrap': [True],
        'oob_score': [True],
        }   

        vr_scorer = make_scorer(violation_ratio, greater_is_better=False)


        print("----<<   GRID SEARCH   >>----")
        grid = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1),
                        param_grid, cv=5, n_jobs=-1, scoring=vr_scorer)
        grid.fit(train_X, train_y)
        return grid.best_estimator_ , test_X, test_y
    else:
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        model.fit(train_X, train_y)
        return model, test_X, test_y


def svm_model(train_X, train_y, grid_search=False):
    # Split the data into training and testing sets
    print("----<<   SPLITTING DATA   >>----")
    train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)
    
    print("----<<   TRAINING MODEL   >>----")
    
    if grid_search:
        param_grid = {
            'C': [0.1, 1, 10, 100, 1000],
            'epsilon': [0.0001, 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': ['scale', 'auto']
        }

        vr_scorer = make_scorer(violation_ratio, greater_is_better=False)

        print("----<<   GRID SEARCH   >>----")
        # SVR grid search
        grid = GridSearchCV(SVR(), param_grid, cv=5, n_jobs=-1, scoring=vr_scorer)
        grid.fit(train_X, train_y)
        
        print("Best parameters found: ", grid.best_params_)
        
        return grid.best_estimator_, test_X, test_y
    else:
        model = SVR()
        model.fit(train_X, train_y)
        return model, test_X, test_y


def xgboost_model(train_X, train_y):
    # Split the data into training and testing sets
    print("----<<   SPLITTING DATA   >>----")
    train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)
    
    print("----<<   TRAINING MODEL   >>----")
    model = xgb.XGBRegressor(n_estimators=100, max_depth=20, learning_rate=0.1, objective='reg:squarederror', random_state=42)
    model.fit(train_X, train_y)
    return model, test_X, test_y


def linear_regression_model(train_X, train_y):
    # Split the data into training and testing sets
    print("----<<   SPLITTING DATA   >>----")
    train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, test_size=0.2, random_state=42)

    print("----<<   TRAINING MODEL   >>----")
    model = LinearRegression()
    model.fit(train_X, train_y)
    return model, test_X, test_y



if __name__ == "__main__":


    # Argparse
    parser = argparse.ArgumentParser()
    # Model selection: 0: svm, 1: random_forest, 2: xgboost
    parser.add_argument('--input', type=str, default=None, help='Input file', required=True)
    parser.add_argument('--model', type=int, default=1, help='Model selection: 0: svm, 1: random_forest, 2: xgboost, 3: linear_regression')
    parser.add_argument('-o', '--output', type=str, default='model.cpp', help='Output file')
    parser.add_argument('--grid-search', action='store_true', help='Enable grid search')
    args = parser.parse_args()
    model = args.model
    output_file = args.output
    input_file = args.input
    if input_file is None:
        print("Error: Input file is required")
        exit(1)

    train_X, train_y = prepare_train_data(input_file)
    #print("train_X: ", train_X)
    print("train_y: ", train_y)

    # Use different models here
    if model == 0:
        print("----<<   SVM MODEL   >>----")
        best_model, test_X, test_y = svm_model(train_X, train_y, args.grid_search)
    elif model == 1:
        print("----<<   RANDOM FOREST MODEL   >>----")
        best_model, test_X, test_y = random_forest_model(train_X, train_y, args.grid_search)
    elif model == 2:
        print("----<<   XGBOOST MODEL   >>----")
        best_model, test_X, test_y = xgboost_model(train_X, train_y)
    elif model == 3:
        print("----<<   LINEAR REGRESSION MODEL   >>----")
        best_model, test_X, test_y = linear_regression_model(train_X, train_y)

    # best_model, test_X, test_y = random_forest_model(train_X, train_y)

    print("----<<   BEST MODEL   >>----")
    # Predict the test set
    test_y_pred = np.round(best_model.predict(test_X)).astype(int)
    # print("test_y: ", test_y)
    # print("test_y_pred: ", test_y_pred)

    # Calculate the mean squared error
    mse = mean_squared_error(test_y, test_y_pred)
    print("----<<   TEST RESULTS   >>----")
    print(f"Violation Ratio (VR): {100*violation_ratio(test_y, test_y_pred):2.2f} %")
    # print(f"Mean Absolute Percentage Error: {np.mean(np.abs((test_y - test_y_pred) / test_y)) * 100:2.2f}")
    print(f"R-squared: {r2_score(test_y, test_y_pred):2.2f}")
    print(f"Mean Squared Error: {mse:2.2f}")
    print(f"Root Mean Squared Error: {np.sqrt(mse):2.2f}")

    # Convert the model to a Cpp source code
    print("----<<   CONVERTING MODEL TO CPP   >>----")
    cpp_code = m2c.export_to_c(best_model)
    open(output_file, 'w').write(cpp_code)

