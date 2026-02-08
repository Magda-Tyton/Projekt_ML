import numpy as np
from sklearn import linear_model, metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import joblib

baseline = lambda y:  np.ones(len(y))*np.average(y)

def best_model_regressor_fp(X_train, y_train, X_subtest, y_subtest, X_test, f=lambda x: x, g=lambda x: x):
    models = {
        # "Linear (double split)": linear_model.LinearRegression(),
        # "Ridge": linear_model.Ridge(alpha=0.1),
        # "Lasso": linear_model.Lasso(alpha=0.01),
        # "kNN": KNeighborsRegressor(n_neighbors=50, weights="distance"),
        # "ElasticNet": linear_model.ElasticNet(alpha=0.1, l1_ratio=0.5),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, max_depth=20),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, f(y_train))
        y_val_pred = g(model.predict(X_subtest))
        mse_val_best = metrics.mean_squared_error(y_subtest, y_val_pred)
        results[name] = mse_val_best
        print(f"{name}: MSE walidacja = {mse_val_best:.4f}")

    best_model_name = min(results, key=results.get)
    print(f"Najlepszy model: {best_model_name}")

    y_pred = models[best_model_name].predict(X_test)
    return y_pred

def best_model_log_regressor_fp(X_train, y_train, X_subtest, y_subtest, X_test):
    return best_model_regressor_fp(X_train, y_train, X_subtest, y_subtest, X_test, np.log1p, np.expm1)