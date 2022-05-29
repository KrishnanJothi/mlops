import argparse
import os
import pickle
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run(data_path):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))

    mlflow.sklearn.autolog()
    
    with mlflow.start_run():
        mlflow.set_tag("module-2","experiment-tracking")
        #mlflow.log_param("train_val_data_path",data_path)
        
        rf_reg_max_depth = 10
        #mlflow.log_param("RandomForest_regressor_max_depth",rf_reg_max_depth)
        
        rf = RandomForestRegressor(max_depth=rf_reg_max_depth, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_valid)

        rmse = mean_squared_error(y_valid, y_pred, squared=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="/home/Krishnan/mlops-zoomcamp/02-experiment-tracking/homework/output/",
        help="the location where the processed NYC taxi trip data was saved."
    )
    args = parser.parse_args()
    
    MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("nyc-taxi-experiment")

    run(args.data_path)
