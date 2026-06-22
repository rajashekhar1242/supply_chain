import pandas as pd
import numpy as np
from ingestion.utils import log
from supabase import create_client


# ------------------------------------------------------------
# Generate Demand Features (Supabase + Pagination Version)
# ------------------------------------------------------------
def generate_demand_features(supabase):

    try:
        log.info("Starting Demand Feature Generation (Supabase)...")

        # -------------------------------------------------
        # Pagination Fetch (Important for 31k+ rows)
        # -------------------------------------------------
        all_data = []
        page_size = 1000
        start = 0

        while True:
            response = (
                supabase
                .table("fact_order_line")
                .select("order_placement_date, product_id, delivery_qty")
                .range(start, start + page_size - 1)
                .execute()
            )

            batch = response.data

            if not batch:
                break

            all_data.extend(batch)
            log.info(f"Fetched rows {start} to {start + page_size}")
            start += page_size

        if not all_data:
            log.warning("No data found in fact_order_line")
            return

        df = pd.DataFrame(all_data)

        df.rename(columns={
            "order_placement_date": "date",
            "delivery_qty": "units_sold"
        }, inplace=True)

        required_cols = ["product_id", "date", "units_sold"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df["date"] = pd.to_datetime(df["date"])
        df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce").fillna(0)

        df = df.sort_values(["product_id", "date"])

        feature_rows = []

        for product_id, group in df.groupby("product_id"):

            group = (
                group.groupby("date", as_index=False)["units_sold"]
                .sum()
                .sort_values("date")
            )

            group.set_index("date", inplace=True)

            full_index = pd.date_range(
                start=group.index.min(),
                end=group.index.max(),
                freq="D"
            )

            group = group.reindex(full_index)

            group["units_sold"] = group["units_sold"].fillna(0)
            group["product_id"] = product_id
            group.index.name = "date"

            group["rolling_7d_mean"] = group["units_sold"].rolling(7, min_periods=1).mean()
            group["rolling_30d_mean"] = group["units_sold"].rolling(30, min_periods=1).mean()
            group["rolling_30d_std"] = group["units_sold"].rolling(30, min_periods=1).std().fillna(0)

            group["volatility_index"] = np.where(
                group["rolling_30d_mean"] == 0,
                0,
                group["rolling_30d_std"] / group["rolling_30d_mean"]
            )

            group["weekly_growth_pct"] = group["units_sold"].pct_change(7)

            group["demand_spike_flag"] = (
                group["units_sold"] >
                group["rolling_30d_mean"] +
                2 * group["rolling_30d_std"]
            ).astype(int)

            group.reset_index(inplace=True)
            feature_rows.append(group)

        df_features = pd.concat(feature_rows, ignore_index=True)

        df_features = df_features[[
            "date",
            "product_id",
            "units_sold",
            "rolling_7d_mean",
            "rolling_30d_mean",
            "rolling_30d_std",
            "volatility_index",
            "weekly_growth_pct",
            "demand_spike_flag"
        ]]

        # -------------------------------------------------
        # Fix Timestamp JSON Issue
        # -------------------------------------------------
        df_features["date"] = pd.to_datetime(df_features["date"]).dt.strftime("%Y-%m-%d")

        # -------------------------------------------------
        # 🔥 VERY IMPORTANT: Clean inf / -inf / NaN
        # -------------------------------------------------
        df_features.replace([np.inf, -np.inf], 0, inplace=True)
        df_features.fillna(0, inplace=True)

        # -------------------------------------------------
        # Clear old data safely
        # -------------------------------------------------
        supabase.table("demand_features") \
            .delete() \
            .not_.is_("product_id", None) \
            .execute()

        # -------------------------------------------------
        # Insert in batches
        # -------------------------------------------------
        records = df_features.to_dict(orient="records")

        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            supabase.table("demand_features").insert(batch).execute()

        log.info("Demand Features updated successfully.")

    except Exception as e:
        log.error(f"Demand Engine failed: {e}", exc_info=True)
        raise


# -------------------------------------------------
# 🔥 AIRFLOW ENTRY POINT
# -------------------------------------------------
def run(**context):
    from ingestion.config import load_settings

    log.info("Initializing Supabase for Demand Engine...")

    settings = load_settings(env="dev")

    supabase = create_client(
        settings.supabase_url,
        settings.supabase_key
    )

    generate_demand_features(supabase)


if __name__ == "__main__":
    run()