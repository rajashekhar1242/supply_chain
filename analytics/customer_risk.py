import pandas as pd
import lightgbm as lgb
from datetime import date, datetime, timedelta
from supabase import create_client, Client
from ingestion.config import load_settings
from ingestion.utils import log
import sys
import logging
import time
import httpx

# ----------------------------------------------------------------------------
# 1. UTF-8 Console Logging Setup
# ----------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    encoding="utf-8",
)

# ----------------------------------------------------------------------------
# 2. Load Settings & Initialize Supabase
# ----------------------------------------------------------------------------
settings = load_settings()
supabase: Client = create_client(settings.supabase_url, settings.supabase_service_role_key)

# ----------------------------------------------------------------------------
# 3. Supabase Fetch with Retry (optional date filter)
# ----------------------------------------------------------------------------
def fetch_with_retry(table_name: str, select_fields: str, start_date: str = None, date_column: str = None, retries=3, delay=2):
    for attempt in range(retries):
        try:
            query = supabase.table(table_name).select(select_fields)
            if start_date and date_column:
                query = query.gte(date_column, start_date)
            return query.execute()
        except (httpx.RemoteProtocolError, httpx.RequestError) as e:
            log.warning(f"Attempt {attempt+1} failed fetching {table_name}: {e}")
            time.sleep(delay)
    raise ConnectionError(f"Failed to fetch data from {table_name} after {retries} attempts")

# ----------------------------------------------------------------------------
# 4. Fetch KPI Data (Last 180 Days) + Time-based Features
# ----------------------------------------------------------------------------
def fetch_kpi_data() -> pd.DataFrame:
    log.info("Calculating real OTIF % from last 180 days of order data...")

    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")

    # Fetch orders with retry
    orders = fetch_with_retry(
        "fact_order_line",
        "customer_id, order_placement_date, \"On Time\", \"In Full\", \"On Time In Full\"",
        start_date=start_date,
        date_column="order_placement_date"
    )

    df = pd.DataFrame(orders.data)
    if df.empty:
        log.warning("No order data found.")
        return pd.DataFrame()

    # Convert and clean
    df["order_placement_date"] = pd.to_datetime(df["order_placement_date"], errors="coerce")
    for col in ["On Time", "In Full", "On Time In Full"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --- Time-based features ---
    df["order_day_of_week"] = df["order_placement_date"].dt.dayofweek
    df["order_month"] = df["order_placement_date"].dt.month

    # Days since last order per customer
    df = df.sort_values(["customer_id", "order_placement_date"])
    df["days_since_last_order"] = df.groupby("customer_id")["order_placement_date"].diff().dt.days.fillna(0)

    # --- Rolling OTIF trend feature ---
    df["otif_30d_avg"] = (
        df.groupby("customer_id")["On Time In Full"]
        .transform(lambda x: x.rolling(window=30, min_periods=5).mean() * 100)
        .fillna(0)
    )
    df["recent_otif_trend"] = df["otif_30d_avg"].apply(
        lambda x: -1 if x < 50 else (1 if x > 80 else 0)
    )

    # --- Aggregate by customer ---
    df_agg = df.groupby("customer_id").agg(
        total_orders=("customer_id", "count"),
        on_time_orders=("On Time", "sum"),
        in_full_orders=("In Full", "sum"),
        otif_orders=("On Time In Full", "sum"),
        recent_otif_trend=("recent_otif_trend", "mean"),
        avg_days_since_last_order=("days_since_last_order", "mean"),
        avg_order_day_of_week=("order_day_of_week", "mean"),
        avg_order_month=("order_month", "mean")
    ).reset_index()

    df_agg["on_time_pct"] = (df_agg["on_time_orders"] / df_agg["total_orders"] * 100).round(2)
    df_agg["in_full_pct"] = (df_agg["in_full_orders"] / df_agg["total_orders"] * 100).round(2)
    df_agg["otif_pct"] = (df_agg["otif_orders"] / df_agg["total_orders"] * 100).round(2)

    # Fetch targets (no date filter)
    targets = fetch_with_retry("dim_targets_order", "*")
    df_targets = pd.DataFrame(targets.data)
    if not df_targets.empty:
        df_agg = df_agg.merge(df_targets, on="customer_id", how="left")

    # Fetch customer metadata (no date filter)
    customers = fetch_with_retry("dim_customers", "customer_id, city")
    df_customers = pd.DataFrame(customers.data)
    if not df_customers.empty:
        df_agg = df_agg.merge(df_customers, on="customer_id", how="left")
        df_agg["city_encoded"] = pd.Categorical(df_agg["city"]).codes
    else:
        df_agg["city_encoded"] = 0

    log.info(f"Calculated OTIF, trends, and time-based features for {len(df_agg)} customers.")
    return df_agg

# ----------------------------------------------------------------------------
# 5. Prepare Dataset
# ----------------------------------------------------------------------------
def prepare_dataset(df: pd.DataFrame):
    log.info("Preparing dataset with engineered features...")

    df.rename(
        columns={
            "otif_target%": "target_otif",
            "ontime_target%": "target_ontime",
            "infull_target%": "target_infull",
        },
        inplace=True,
    )

    for col in ["target_otif", "target_ontime", "target_infull"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(70.0)

    df["label"] = (df["otif_pct"] < 70).astype(int)

    feature_cols = [
        "otif_pct",
        "on_time_pct",
        "in_full_pct",
        "total_orders",
        "target_otif",
        "target_ontime",
        "target_infull",
        "recent_otif_trend",
        "city_encoded",
        "avg_days_since_last_order",
        "avg_order_day_of_week",
        "avg_order_month"
    ]

    df[feature_cols] = df[feature_cols].fillna(0)
    log.info(f"Dataset prepared with {len(df)} rows and features: {feature_cols}")
    return df, feature_cols

# ----------------------------------------------------------------------------
# 6. Model + Business Rule Hybrid
# ----------------------------------------------------------------------------
def train_and_predict(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    log.info("Generating predictions...")

    # Step 1: Compute risk score (model or fallback)
    if len(df) >= 10 and df["label"].nunique() > 1:
        try:
            model = lgb.LGBMClassifier(
                n_estimators=50,
                learning_rate=0.1,
                num_leaves=31,
                min_data_in_leaf=1,
                random_state=42,
            )
            model.fit(df[feature_cols], df["label"])
            df["risk_score"] = model.predict_proba(df[feature_cols])[:, 1]
        except Exception as e:
            log.warning(f"Model failed, using fallback: {e}")
            df["risk_score"] = (70 - df["otif_pct"]) / 70
    else:
        df["risk_score"] = (70 - df["otif_pct"]) / 70

    # Step 2: Assign label based on risk score (0.5 threshold)
    df["risk_label"] = df["risk_score"].apply(lambda x: "At Risk" if x > 0.5 else "Safe")

    return df
# ----------------------------------------------------------------------------
# 7. Store Predictions
# ----------------------------------------------------------------------------
def store_predictions(df: pd.DataFrame):
    log.info("Uploading risk predictions to Supabase...")

    if df.empty:
        log.warning("No data to insert.")
        return

    records = [
        {
            "customer_id": int(row["customer_id"]),
            "prediction_date": date.today().isoformat(),
            "risk_score": float(row.get("risk_score", 0.0)),
            "risk_label": str(row.get("risk_label", "Safe")),
        }
        for _, row in df.iterrows()
    ]

    if records:
        supabase.table("kpi_risk_predictions").insert(records).execute()
        log.info(f"Successfully inserted {len(records)} predictions.")

# ----------------------------------------------------------------------------
# 8. Run Pipeline
# ----------------------------------------------------------------------------
def run_customer_risk_pipeline():
    log.info("Starting Customer KPI Risk Pipeline...")

    df = fetch_kpi_data()
    if df.empty:
        log.warning("No data available.")
        return

    df_prepared, features = prepare_dataset(df)
    df_predicted = train_and_predict(df_prepared, features)
    store_predictions(df_predicted)

    log.info("Customer risk pipeline completed successfully.")

# ----------------------------------------------------------------------------
# 9. Execute
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    run_customer_risk_pipeline()
