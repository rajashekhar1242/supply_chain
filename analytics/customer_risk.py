import pandas as pd
from datetime import datetime, timedelta, date
from ingestion.utils import log
import numpy as np

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from supabase import create_client


# ------------------------------------------------------------
# 1️⃣ Fetch KPI Data (Using Supabase Client)
# ------------------------------------------------------------
def fetch_kpi_data(supabase):

    log.info("Fetching OTIF data from Supabase...")

    start_date = (datetime.now() - timedelta(days=3650)).strftime("%Y-%m-%d")

    response = (
        supabase
        .table("fact_order_line")
        .select("customer_id, order_placement_date, on_time, in_full, on_time_in_full")
        .gte("order_placement_date", start_date)
        .execute()
    )

    data = response.data

    if not data:
        log.warning("No order data found.")
        return pd.DataFrame()

    df = pd.DataFrame(data)

    df["order_placement_date"] = pd.to_datetime(df["order_placement_date"])

    df[["on_time", "in_full", "on_time_in_full"]] = (
        df[["on_time", "in_full", "on_time_in_full"]]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    df = df.sort_values(["customer_id", "order_placement_date"])

    df["days_since_last_order"] = (
        df.groupby("customer_id")["order_placement_date"]
        .diff()
        .dt.days
        .fillna(0)
    )

    df["otif_30d_avg"] = (
        df.groupby("customer_id")["on_time_in_full"]
        .transform(lambda x: x.rolling(30, min_periods=5).mean() * 100)
        .fillna(0)
    )

    df_agg = df.groupby("customer_id").agg(
        total_orders=("customer_id", "count"),
        on_time_orders=("on_time", "sum"),
        in_full_orders=("in_full", "sum"),
        otif_orders=("on_time_in_full", "sum"),
        avg_days_since_last_order=("days_since_last_order", "mean"),
        otif_30d_avg=("otif_30d_avg", "mean")
    ).reset_index()

    df_agg["on_time_pct"] = df_agg["on_time_orders"] / df_agg["total_orders"] * 100
    df_agg["in_full_pct"] = df_agg["in_full_orders"] / df_agg["total_orders"] * 100
    df_agg["otif_pct"] = df_agg["otif_orders"] / df_agg["total_orders"] * 100

    return df_agg


# ------------------------------------------------------------
# 2️⃣ Risk Scoring
# ------------------------------------------------------------
def calculate_risk_score(df):

    feature_cols = [
        "otif_pct",
        "on_time_pct",
        "in_full_pct",
        "avg_days_since_last_order",
        "otif_30d_avg"
    ]

    X = df[feature_cols].fillna(0)

    threshold = df["otif_pct"].quantile(0.3)
    df["risk_class"] = (df["otif_pct"] < threshold).astype(int)

    if df["risk_class"].nunique() < 2:

        log.warning("Insufficient label diversity — using fallback scoring.")

        df["risk_score"] = (
            (df["otif_pct"].max() - df["otif_pct"]) /
            (df["otif_pct"].max() - df["otif_pct"].min() + 1e-6)
        )

    else:

        ensemble_model = VotingClassifier(
            estimators=[
                ("lr", Pipeline([
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=500))
                ])),
                ("rf", RandomForestClassifier(
                    n_estimators=120,
                    max_depth=4,
                    random_state=42
                )),
                ("gb", GradientBoostingClassifier(
                    n_estimators=120,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                ))
            ],
            voting="soft"
        )

        ensemble_model.fit(X, df["risk_class"])
        df["risk_score"] = ensemble_model.predict_proba(X)[:, 1]

    # Normalize score 0–1
    df["risk_score"] = (
        (df["risk_score"] - df["risk_score"].min()) /
        (df["risk_score"].max() - df["risk_score"].min() + 1e-6)
    )

    df["risk_label"] = pd.qcut(
        df["risk_score"],
        q=3,
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )

    return df


# ------------------------------------------------------------
# 3️⃣ Store Predictions
# ------------------------------------------------------------
def store_predictions(supabase, df):

    records = df[[
        "customer_id",
        "risk_score",
        "risk_label"
    ]].copy()

    records["prediction_date"] = str(date.today())

    supabase.table("kpi_risk_predictions") \
        .insert(records.to_dict(orient="records")) \
        .execute()

    log.info("Customer risk predictions stored (historical mode).")


# ------------------------------------------------------------
# Core Pipeline
# ------------------------------------------------------------
def run_customer_risk_pipeline(supabase):

    log.info("Starting Customer Risk Engine...")

    df = fetch_kpi_data(supabase)

    if df.empty:
        log.warning("No KPI data available — skipping risk scoring.")
        return

    df_scored = calculate_risk_score(df)
    store_predictions(supabase, df_scored)

    log.info("Customer risk pipeline completed successfully.")


# ------------------------------------------------------------
# 🔥 AIRFLOW ENTRY POINT
# ------------------------------------------------------------
def run(**context):

    try:
        from ingestion.config import load_settings

        log.info("Initializing Supabase Client...")

        settings = load_settings(env="dev")

        supabase = create_client(
            settings.supabase_url,
            settings.supabase_key   # ✅ FIXED
        )

        run_customer_risk_pipeline(supabase)

    except Exception as e:
        log.error(f"Customer Risk Engine failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run()