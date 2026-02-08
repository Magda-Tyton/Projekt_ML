import numpy as np
from sklearn import linear_model, metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

baseline = lambda y:  np.ones(len(y))*np.average(y)

models = {
    "Linear (double split)": linear_model.LinearRegression(),
    "Ridge": linear_model.Ridge(alpha=0.1),
    "Lasso": linear_model.Lasso(alpha=0.01),
    "kNN": KNeighborsRegressor(n_neighbors=50, weights="distance"),
    "ElasticNet": linear_model.ElasticNet(alpha=0.1, l1_ratio=0.5),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, max_depth=20),
}

def best_model_regressor_fp(X_train, y_train, X_test, y_test, f=lambda x: x, g=lambda x: x):

    results = {}
    for name, model in models.items():

        model.fit(X_train, f(y_train.values.ravel()))
        

        y_test_pred = g(model.predict(X_test))
        
        mse_test = metrics.mean_squared_error(y_test, y_test_pred)
        results[name] = mse_test
        print(f"{name}: MSE test = {mse_test:.4f}")


    best_name = min(results, key=results.get)
    print(f"\nNajlepszy model (skala log.): {best_name}")

    y_train_f = f(y_train.values.ravel())
    best_model = models[best_name]
    best_model.fit(X_train, y_train_f)

    return g(best_model.predict(X_test))

def best_model_log_regressor_fp(X_train, y_train, X_subtest, y_subtest, X_test):
    return best_model_regressor_fp(X_train, y_train, X_subtest, y_subtest, X_test, np.log1p, np.expm1)

def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    if 'price' not in df.columns:
        raise ValueError('Target column "price" not found in CSV')
    # Ensure price is numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # Candidate numeric features to use (if present)
    feature_candidates = [
        'mileage_km', 'power_kw', 'power_hp', 'cylinders_volume_cc',
        'weight_kg', 'car_age', 'nr_seats', 'nr_doors',
        'co2_emission_grper_km', 'fuel_cons_comb_l100_km'
    ]

    features = []
    for c in feature_candidates:
        if c in df.columns:
            # coerce to numeric in case of bad parsing
            df[c] = pd.to_numeric(df[c], errors='coerce')
            features.append(c)

    if len(features) == 0:
        # Fallback to any numeric columns except price
        X = df.select_dtypes(include=[np.number]).copy()
        if 'price' in X.columns:
            X = X.drop(columns=['price'])
    else:
        X = df[features].copy()

    # Drop rows with missing target
    df = df.dropna(subset=['price'])
    X = X.loc[df.index]

    # Impute numeric features with median (less destructive than dropna)
    X = X.fillna(X.median())

    y = df['price']

    if X.shape[0] == 0:
        raise ValueError('No training rows available after preprocessing. Check CSV parsing and column names.')

    return X, y


def train_and_evaluate(X, y, model_out_path=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"GradientBoostingRegressor results:\n  MSE: {mse:.4f}\n  RMSE: {rmse:.4f}\n  R2: {r2:.4f}")

    if model_out_path:
        os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
        joblib.dump(model, model_out_path)
        print(f"Saved model to {model_out_path}")

    return model, mse, rmse, r2


def main():
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'data_cleaned.csv')
    model_out = os.path.join(os.path.dirname(__file__), 'gb_model.joblib')
    print(f"Loading data from {csv_path}")
    X, y = load_and_prepare(csv_path)
    print(f"Data shape after numeric-selection and dropna: X={X.shape}, y={y.shape}")
    train_and_evaluate(X, y, model_out)


if __name__ == '__main__':
    main()

