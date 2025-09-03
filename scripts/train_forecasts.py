
#!/usr/bin/env python3
# Train baseline demand forecasting models and compare accuracy.
# Uses lag features + calendar features with scikit-learn models.
from pathlib import Path
import argparse, pandas as pd, numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def make_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['store_id','sku_id','date'])
    for lag in [1,7,14,28]:
        df[f'lag_{lag}'] = df.groupby(['store_id','sku_id'])['sales_qty'].shift(lag)
    df['roll7'] = df.groupby(['store_id','sku_id'])['sales_qty'].shift(1).rolling(7).mean()
    df['roll28'] = df.groupby(['store_id','sku_id'])['sales_qty'].shift(1).rolling(28).mean()
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['year'] = df['date'].dt.year.astype(int)
    return df

def main(args):
    df = pd.read_csv(args.input_csv)
    df = make_features(df).dropna()
    features = ['dow','month','promo_flag','price','temperature_F','precip_flag',
                'lag_1','lag_7','lag_14','lag_28','roll7','roll28','dayofyear','weekofyear','year']
    X = df[features].values
    y = df['sales_qty'].values
    groups = df[['store_id','sku_id']].astype(str).agg('-'.join, axis=1).values
    gkf = GroupKFold(n_splits=5)
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42)
    }
    results = {}
    for name, model in models.items():
        maes = []
        for train_idx, test_idx in gkf.split(X, y, groups):
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx]).clip(0)
            maes.append(mean_absolute_error(y[test_idx], y_pred))
        results[name] = float(np.mean(maes))
    # Naive last-week baseline
    maes = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        y_pred = df.iloc[test_idx]['lag_7'].fillna(df['lag_1'].median()).values
        maes.append(mean_absolute_error(y[test_idx], y_pred))
    results['Naive_lag7'] = float(np.mean(maes))
    out = pd.DataFrame({'model': list(results.keys()), 'CV_MAE': list(results.values())}).sort_values('CV_MAE')
    Path('outputs').mkdir(exist_ok=True, parents=True)
    out.to_csv(args.output_csv, index=False)
    print(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', default='data/retail_sales.csv')
    parser.add_argument('--output_csv', default='outputs/model_cv_mae.csv')
    args = parser.parse_args()
    Path('outputs').mkdir(exist_ok=True, parents=True)
    main(args)
