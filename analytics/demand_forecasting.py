import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from ingestion.utils import log
from datetime import datetime
from supabase import create_client


def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100


# ------------------------------------------------------------
# CORE FORECAST ENGINE (SUPABASE VERSION)
# ------------------------------------------------------------
def generate_advanced_demand_forecast(supabase, forecast_horizon=7):

    try:
        log.info("Starting Advanced Global Demand Forecasting...")

        # -------------------------------------------------
        # Fetch demand_features with pagination
        # -------------------------------------------------
        all_data = []
        page_size = 1000
        start = 0

        while True:
            response = (
                supabase
                .table("demand_features")
                .select("*")
                .range(start, start + page_size - 1)
                .execute()
            )

            batch = response.data

            if not batch:
                break

            all_data.extend(batch)
            start += page_size

        if not all_data:
            log.warning("No data in demand_features.")
            return

        df = pd.DataFrame(all_data)

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["product_id", "date"])

        # -------------------------------------------------
        # FEATURE ENGINEERING
        # -------------------------------------------------
        df["lag_1"] = df.groupby("product_id")["units_sold"].shift(1)
        df["lag_7"] = df.groupby("product_id")["units_sold"].shift(7)
        df["lag_14"] = df.groupby("product_id")["units_sold"].shift(14)

        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["day_of_month"] = df["date"].dt.day

        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)

        feature_cols = [
            "product_id",
            "lag_1",
            "lag_7",
            "lag_14",
            "rolling_7d_mean",
            "rolling_30d_mean",
            "rolling_30d_std",
            "volatility_index",
            "weekly_growth_pct",
            "demand_spike_flag",
            "day_of_week",
            "month",
            "day_of_month"
        ]

        categorical_features = ["product_id"]

        # -------------------------------------------------
        # TIME-SERIES SPLIT
        # -------------------------------------------------
        unique_dates = sorted(df["date"].unique())
        split_point = int(len(unique_dates) * 0.8)

        train_dates = unique_dates[:split_point]
        test_dates = unique_dates[split_point:]

        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"].isin(test_dates)]

        X_train = train_df[feature_cols]
        y_train = train_df["units_sold"]

        X_test = test_df[feature_cols]
        y_test = test_df["units_sold"]

        model = CatBoostRegressor(
            iterations=800,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            verbose=False
        )

        model.fit(X_train, y_train, cat_features=categorical_features)

        preds = model.predict(X_test)

        mape = calculate_mape(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        log.info(f"Validation MAPE: {mape:.2f}%")
        log.info(f"Validation RMSE: {rmse:.2f}")

        # -------------------------------------------------
        # Store Metrics
        # -------------------------------------------------
        metrics_record = {
            "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fold": 1,
            "train_end_date": str(train_dates[-1]),
            "test_start_date": str(test_dates[0]),
            "test_end_date": str(test_dates[-1]),
            "mape": float(mape),
            "rmse": float(rmse)
        }

        supabase.table("demand_model_metrics").insert(metrics_record).execute()

        # -------------------------------------------------
        # Train Final Model on Full Data
        # -------------------------------------------------
        final_model = CatBoostRegressor(
            iterations=800,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            verbose=False
        )

        final_model.fit(
            df[feature_cols],
            df["units_sold"],
            cat_features=categorical_features
        )

        log.info("Final model trained on full dataset.")

        # -------------------------------------------------
        # Recursive Forecast
        # -------------------------------------------------
        forecast_rows = []
        last_dates = df.groupby("product_id")["date"].max()

        for product_id in df["product_id"].unique():

            product_history = df[df["product_id"] == product_id]

            if len(product_history) < 30:
                continue

            last_row = product_history.iloc[-1:].copy()
            current_date = last_dates[product_id]

            for _ in range(forecast_horizon):

                current_date += pd.Timedelta(days=1)

                last_row["day_of_week"] = current_date.dayofweek
                last_row["month"] = current_date.month
                last_row["day_of_month"] = current_date.day

                pred = final_model.predict(last_row[feature_cols])[0]
                pred = float(max(pred, 0))

                forecast_rows.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "product_id": int(product_id),
                    "predicted_units": pred
                })

                last_row["lag_14"] = last_row["lag_7"]
                last_row["lag_7"] = last_row["lag_1"]
                last_row["lag_1"] = pred

        # -------------------------------------------------
        # Clear old forecasts safely
        # -------------------------------------------------
        supabase.table("demand_forecast") \
            .delete() \
            .not_.is_("product_id", None) \
            .execute()

        # -------------------------------------------------
        # Insert new forecasts in batches
        # -------------------------------------------------
        batch_size = 500
        for i in range(0, len(forecast_rows), batch_size):
            batch = forecast_rows[i:i + batch_size]
            supabase.table("demand_forecast").insert(batch).execute()

        log.info(" Demand Forecast updated successfully.")

    except Exception as e:
        log.error(f" Forecasting failed: {e}", exc_info=True)
        raise


# ------------------------------------------------------------
# AIRFLOW ENTRY
# ------------------------------------------------------------
def run(**context):

    from ingestion.config import load_settings

    log.info("Initializing Supabase for Advanced Demand Forecasting...")

    settings = load_settings(env="dev")

    supabase = create_client(
        settings.supabase_url,
        settings.supabase_key
    )

    generate_advanced_demand_forecast(supabase)


if __name__ == "__main__":
    run()