from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import pandas as pd

from compile_data import *
import argparse

# Read CSV file
csv_file = '/Users/archit/Documents/Watch_hand_wash_classify/balanced_features.csv'

# Save ARFF file
arff_file = '/Users/archit/Documents/Watch_hand_wash_classify/arff_data/balanced_features.arff'

def read_csv(filename) -> pd.DataFrame:
    # Read CSV file into a DataFrame
    data = pd.read_csv(filename)
    if data.empty:
        print("No data found")
        return None
    return data


def get_data(args):
    df = read_csv(args.csv_path)
    if df is None:
        return None
    
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    y = y.map({'indoor walk': 0, 'outdoor walk': 1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    return X_train, X_test, y_train, y_test

def classify(args) -> float:
    X_train, X_test, y_train, y_test = get_data(args)
    if X_train is None:
        return -1

    if args.classifier == 1:
        classifier = RandomForestClassifier()
        param_grid = {
            'n_estimators': [5, 10, 15, 20, 50, 100],
            'max_depth': [2, 5, 7, 9, 10]
        }

        grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=10)
        grid_search.fit(X_train, y_train)
        classifier = grid_search.best_estimator_
        # print("Random Forest best parameters: ", grid_search.best_params_)

        return classifier.score(X_test, y_test) * 100

    elif args.classifier == 2:
        classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        data_dmatrix = xgb.DMatrix(X_train, label=y_train)
        def fit(x):
            params = {'objective':'binary:logistic',
                    'eval_metric':'mlogloss',
                    'eta':x[0],
                    'subsample':x[1]}
            
            xgb_cv = xgb.cv(dtrain=data_dmatrix, params=params, 
                nfold=10, metrics = 'logloss',seed=42)
            
            return xgb_cv[-1:].values[0]

        grid = pd.DataFrame({'eta':[0.01,0.05,0.1]*2,
            'subsample':np.repeat([0.1,0.3],3)})
        
        grid[['train-logloss-mean','train-logloss-std',
            'test-logloss-mean','test-logloss-std']] = grid.apply(fit, axis=1, result_type='expand')

        min_loss = grid['test-logloss-mean'].min()
        best_params = grid[grid['test-logloss-mean'] == min_loss]
        best_eta = best_params['eta'].values[0]
        best_subsample = best_params['subsample'].values[0]

        classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', eta=best_eta, subsample=best_subsample)
        classifier.fit(X_train, y_train)

        return classifier.score(X_test, y_test) * 100
    else:
        return -1
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hand wash classification using Weka')
    parser.add_argument('--csv_path', type=str, help='Path to the CSV file', default=csv_file)
    parser.add_argument('--arff_path', type=str, help='Path to the ARFF file', default=arff_file)
    parser.add_argument('--window_size', type=int, help='Window size', default=None)
    parser.add_argument('--add_extra_cols', action='store_true', help='Add extra columns to the data')
    parser.add_argument('--feature_selector', action='store_true', help='Use feature selection')
    parser.add_argument('--classifier', type=int, help='Classifier option: 1 - Random Forest, 2 - XGBoost', default=1)
    args = parser.parse_args()
    
    # if not args.feature_selector:
    #     generate_data(args.window_size, args.add_extra_cols)
    # else:
    #     args.add_extra_cols = True

    df = pd.DataFrame(columns = ['window_size', 'extra_cols', 'classifier', 'accuracy'])
    
    if not args.window_size:
        for window_size in [1000, 2000, 3000, 4000, 6000, 10000]:
            args.window_size = window_size
            generate_data(args.window_size, args.add_extra_cols)
            args.csv_path = f'features_window_size_{args.window_size}_extra_cols_{args.add_extra_cols}_walk.csv'
            accuracy = classify(args)
            classifier = 'Random_Forest' if args.classifier == 1 else 'XGBoost'
            df_ = pd.DataFrame(data={"window_size": args.window_size, "extra_cols": args.add_extra_cols, "classifier": classifier, "accuracy": accuracy}, index=[0])
            print(f"Window size: {args.window_size}, Extra columns: {args.add_extra_cols}, Classifier: {classifier}, Accuracy: {accuracy:.2f}%")
            df = pd.concat([df, df_])

        df.to_csv(f'results_extra_cols_{args.add_extra_cols}_{classifier}.csv', index=False)
    else:
        generate_data(args.window_size, args.add_extra_cols)
        accuracy = classify(args)
        classifier = 'Random Forest' if args.classifier == 1 else 'XGBoost'
        print(f"Window size: {args.window_size}, Extra columns: {args.add_extra_cols}, Classifier: {classifier}, Accuracy: {accuracy:.2f}%")