import pandas as pd
import numpy as np
import random
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
import importlib.util

# Import models module
file_path = os.path.abspath("app/models.py")
spec = importlib.util.spec_from_file_location("models", file_path)
models_module = importlib.util.module_from_spec(spec)
sys.modules["models"] = models_module
spec.loader.exec_module(models_module)


def reconstruct_preprocessing():
    """
    Reconstruct the preprocessing pipeline to get scalers and understand transformations.
    Returns: X_train, X_test, y_train, y_test, scaler, original numeric columns, one-hot mappings
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # Load the original raw data
    df = pd.read_csv(os.path.join(data_dir, 'data_raw.csv'))
    
    # Apply same preprocessing as in preprocessing.ipynb
    # Clean raw data
    df = df.drop_duplicates()
    cols_with_units = ['weight_kg', 'mileage_km']
    for col in cols_with_units:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.replace(r'[^0-9.]', '', regex=True)
                      .replace('', np.nan).astype('float64'))
    
    cols_to_drop = ['mileage_km_raw', 'power_kw', 'price_net', 'price_vat_rate', 'electric_range_city_km']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    # Early feature engineering
    equipment_cols = ['equipment_comfort', 'equipment_entertainment', 'equipment_extra', 'equipment_safety']
    current_year = 2026
    df['registration_year'] = pd.to_datetime(df['registration_date'], errors='coerce').dt.year
    df['production_year'] = df['production_year'].fillna(df['registration_year'])
    df['car_age'] = current_year - df['production_year']
    
    cols_to_drop = ['production_year', 'registration_date', 'registration_year']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    df['ratings_average'] = (df['ratings_average'].astype(str).str.replace(',', '.', regex=False)
                            .str.extract(r'(\d+\.?\d*)')[0].astype(float))
    df['ratings_average'] = df['ratings_average'].fillna(2.5)
    df['ratings_count'] = df['ratings_count'].fillna(0)
    df['ratings_recommend_percentage'] = df['ratings_recommend_percentage'].fillna(0)
    
    C = 2.5
    m = 10
    df['offer_score'] = (C * m + df['ratings_average'] * df['ratings_count']) / (m + df['ratings_count'])
    mask_high_recommend = df['ratings_recommend_percentage'] > 85
    df.loc[mask_high_recommend, 'offer_score'] += 0.5
    df['offer_score'] = df['offer_score'].clip(upper=5.0)
    
    ratings_drop = ['ratings_average', 'ratings_count', 'ratings_recommend_percentage']
    df.drop(columns=[c for c in ratings_drop if c in df.columns], inplace=True)
    
    # Handle missing and unique values
    missing_percent = (df.isnull().sum() / len(df)) * 100
    cols_to_drop = missing_percent[missing_percent > 60].index.tolist()
    df.drop(columns=cols_to_drop, inplace=True)
    
    cols_to_trim = missing_percent[(missing_percent > 0) & (missing_percent < 2)].index.tolist()
    df.dropna(subset=cols_to_trim, inplace=True)
    
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df.drop(columns=constant_cols, inplace=True)
    
    object_cols = df.select_dtypes(include=['object']).columns
    high_card_cols = [c for c in object_cols if df[c].nunique() > 100 and c != 'model']
    df.drop(columns=high_card_cols, inplace=True)
    
    location_cols = ['latitude', 'longitude']
    df.drop(columns=[c for c in location_cols if c in df.columns], inplace=True)
    
    # Handle outliers
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)].copy()
    
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns
    for col in numeric_cols:
        lower_limit = df[col].quantile(0.01)
        upper_limit = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
    
    # Encode and split - store original data before encoding
    df_original = df.copy()
    
    X = df.drop(columns=['price'])
    y = df['price']
    
    SEED = 431
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    
    # Store indices to match with original data
    X_train_original = df_original.loc[X_train.index].drop(columns=['price'])
    X_test_original = df_original.loc[X_test.index].drop(columns=['price'])
    
    # Fill missing values
    zero_fill_cols = ['gears', 'nr_prev_owners']
    for col in zero_fill_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].fillna(0)
            X_test[col] = X_test[col].fillna(0)
    
    numeric_cols = X_train.select_dtypes(include=['int', 'float']).columns
    train_medians = X_train[numeric_cols].median()
    
    X_train[numeric_cols] = X_train[numeric_cols].fillna(train_medians)
    X_test[numeric_cols] = X_test[numeric_cols].fillna(train_medians)
    
    object_cols = X_train.select_dtypes(include=['object']).columns
    for col in object_cols:
        if not X_train[col].mode().empty:
            train_mode = X_train[col].mode()[0]
            X_train[col] = X_train[col].fillna(train_mode)
            X_test[col] = X_test[col].fillna(train_mode)
    
    # Fit scaler BEFORE one-hot encoding
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # One-hot encode
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    return X_train, X_test, y_train, y_test, scaler, numeric_cols, X_train_original, X_test_original


def test_model_with_random_sample():
    """
    Test best_model_regressor_fp with a random sample from X_test.
    Display original scaled values and predicted price.
    """
    print("=" * 40)
    print("TESTING BEST MODEL REGRESSOR WITH RANDOM SAMPLE")
    print("=" * 40)
    
    # Reconstruct preprocessing
    X_train, X_test, y_train, y_test, scaler, numeric_cols, X_train_original, X_test_original = reconstruct_preprocessing()
    
    # Select a random row
    random.seed(42)
    np.random.seed(42)
    random_idx = np.random.randint(0, len(X_test))
    
    print(f"\nRandomly selected test sample index: {random_idx}")
    print(f"Total test samples: {len(X_test)}")
    
    # Get the original unscaled row
    original_row = X_test_original.iloc[random_idx]
    
    print("\n" + "=" * 40)
    print("ORIGINAL VALUES (BEFORE SCALING & ENCODING):")
    print("=" * 40)
    
    # Display numeric values
    print("\nNUMERIC FEATURES (original scale from data_raw):")

    for col in numeric_cols:
        if col in original_row.index:
            val = original_row[col]
            if pd.notna(val):
                print(f"  {col}: {val:.4f}")
    
    # Display categorical values
    print("\nCATEGORICAL FEATURES (decoded categories):")
    categorical_cols = [c for c in original_row.index if c not in numeric_cols]

    for col in categorical_cols:
        if col in original_row.index:
            val = original_row[col]
            if pd.notna(val):
                print(f"  {col}: {val}")
    
    print("\nMOST IMPORTANT FEATURES:")
    cols_chosen = [
        'make',
        'model',
        'body_type',
        'car_age',
        'mileage_km',
        'cylinders_volume_cc',
        'cylinders',
        'fuel_category',
        'power_hp',
        'drive_train',
        'transmission',
        'is_used',
        'is_new',
        'had_accident',
        'offer_score'
    ]

    for col in cols_chosen:
        if col in original_row.index:
            val = original_row[col]
            if pd.notna(val):
                print(f"  {col}: {val}")

    # Get the scaled row for prediction
    scaled_row = X_test.iloc[random_idx].values.reshape(1, -1)
    
    # Train and get predictions from best_model_regressor_fp
    print("\n" + "=" * 40)
    print("TRAINING BEST_MODEL_REGRESSOR_FP...")
    print("=" * 40)
    y_pred = models_module.best_model_regressor_fp(X_train, y_train, X_test, y_test)
    
    # Get prediction for this specific sample
    predicted_price = y_pred[random_idx]
    actual_price = y_test.values[random_idx]
    
    print("\n" + "=" * 40)
    print("PREDICTION RESULTS:")
    print("=" * 40)
    print(f"Predicted Price: €{predicted_price:,.2f}")
    print(f"Actual Price:    €{actual_price:,.2f}")
    error = abs(predicted_price - actual_price)
    error_pct = (error / actual_price) * 100 if actual_price != 0 else 0
    print(f"Absolute Error:  €{error:,.2f} ({error_pct:.2f}%)")
    print("=" * 40)


if __name__ == '__main__':
    def test_model_with_user_input():
        """
        Prompt user for car feature values (original-unencoded columns), preprocess
        them to match encoded `X_train` columns, append to `X_test` and run
        `best_model_regressor_fp` to obtain a prediction for the user-provided car.
        """

        # Reconstruct preprocessing artifacts
        X_train, X_test, y_train, y_test, scaler, numeric_cols, X_train_original, X_test_original = reconstruct_preprocessing()

        original_columns = X_train_original.columns.tolist()

        # Compute medians and modes from original training data for defaults
        train_medians = X_train_original[numeric_cols].median() if len(numeric_cols) > 0 else pd.Series()
        object_cols = X_train_original.select_dtypes(include=['object']).columns.tolist()
        train_modes = {c: (X_train_original[c].mode()[0] if not X_train_original[c].mode().empty else '') for c in object_cols}

        print("\nEnter car details column-by-column. Press Enter to accept the shown default.")

        user_vals = {}
        for col in original_columns:
            try:
                if col in numeric_cols:
                    default = train_medians.get(col, np.nan)
                    prompt = f"{col} (numeric) [default={default:.4f}]: " if pd.notna(default) else f"{col} (numeric): "
                    s = input(prompt).strip()
                    if s == '':
                        val = default
                    else:
                        val = float(s)
                else:
                    # categorical
                    options = X_train_original[col].dropna().unique().tolist()
                    display_opts = options[:20]
                    default = train_modes.get(col, '')
                    prompt = f"{col} (categorical) options={display_opts} [default={default}]: "
                    s = input(prompt).strip()
                    if s == '':
                        val = default
                    else:
                        val = s
            except Exception:
                val = None
            user_vals[col] = val

        user_df = pd.DataFrame([user_vals], columns=original_columns)

        # Align missing numeric cols
        for c in numeric_cols:
            if c not in user_df.columns:
                user_df[c] = train_medians.get(c, 0.0)

        # Fill numeric missing with medians
        if len(numeric_cols) > 0:
            user_df[numeric_cols] = user_df[numeric_cols].fillna(train_medians)

        # Fill categorical missing with train modes
        for c in object_cols:
            if c in user_df.columns:
                user_df[c] = user_df[c].fillna(train_modes.get(c, ''))

        # Scale numeric features using the reconstructed scaler
        if len(numeric_cols) > 0:
            try:
                numeric_block = user_df[numeric_cols].astype(float)
                scaled_block = scaler.transform(numeric_block)
                user_df.loc[:, numeric_cols] = scaled_block
            except Exception as e:
                print("Error scaling numeric inputs:", e)

        # One-hot encode with the same drop_first behavior and align columns
        user_encoded = pd.get_dummies(user_df, drop_first=True)
        user_encoded = user_encoded.reindex(columns=X_train.columns, fill_value=0)

        # Append to X_test and a placeholder y_test (median) so the function can evaluate
        X_test_app = pd.concat([X_test.reset_index(drop=True), user_encoded], ignore_index=True)
        y_placeholder = y_test.median()
        y_test_app = pd.concat([y_test.reset_index(drop=True), pd.Series([y_placeholder])], ignore_index=True)

        print("\nTraining models and predicting for your input (this may take a moment)...\n")
        y_pred_all = models_module.best_model_regressor_fp(X_train, y_train, X_test_app, y_test_app)

        pred_for_user = y_pred_all[-1]
        print("\n" + "=" * 60)
        print("Prediction for provided car details:")
        print(f"Predicted Price: €{pred_for_user:,.2f}")
        print("=" * 60)

    print("Choose test mode:\n 1) Random sample test\n 2) User input test (enter car details)")
    choice = input("Enter 1 or 2 [1]: ").strip() or "1"
    if choice == "2":
        test_model_with_user_input()
    else:
        test_model_with_random_sample()
