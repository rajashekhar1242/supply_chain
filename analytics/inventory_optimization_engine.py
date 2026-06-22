import pandas as pd
import numpy as np
from ingestion.utils import log
from supabase import create_client


# ------------------------------------------------------------
# CORE INVENTORY ENGINE (SUPABASE VERSION)
# ------------------------------------------------------------
def generate_strategic_inventory_policy(supabase, service_level_z=1.65):

    try:
        log.info("Starting Strategic Inventory Optimization...")

        # =====================================================
        # 1️⃣ Load Demand Features (Pagination)
        # =====================================================
        all_features = []
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

            all_features.extend(batch)
            start += page_size

        if not all_features:
            log.warning("No demand features found.")
            return

        df_features = pd.DataFrame(all_features)

        df_features["date"] = pd.to_datetime(df_features["date"])
        df_features["product_id"] = pd.to_numeric(df_features["product_id"], errors="coerce")

        latest_features = (
            df_features
            .sort_values("date")
            .groupby("product_id", as_index=False)
            .tail(1)
        )

        # =====================================================
        # 2️⃣ Cost Data
        # =====================================================
        cost_data = supabase.table("weekly_cost_updates").select("*").execute().data
        df_cost = pd.DataFrame(cost_data)

        if not df_cost.empty:
            df_cost["product_id"] = pd.to_numeric(df_cost["product_id"], errors="coerce")

            latest_cost = (
                df_cost
                .sort_values("week_start_date")
                .groupby("product_id", as_index=False)
                .tail(1)
            )

            latest_cost["holding_cost_per_unit"] = (
                latest_cost["unit_cost"].fillna(0) *
                latest_cost["holding_rate"].fillna(0)
            )
        else:
            latest_cost = pd.DataFrame()

        # =====================================================
        # 3️⃣ Supplier Lead Time
        # =====================================================
        supplier_data = supabase.table("weekly_supplier_updates").select("*").execute().data
        df_supplier = pd.DataFrame(supplier_data)

        if df_supplier.empty:
            avg_lead_time = 7
        else:
            avg_lead_time = (
                df_supplier
                .sort_values("week_start_date")
                .groupby("supplier_id", as_index=False)
                .tail(1)["avg_lead_time_days"]
                .mean()
            )

        if pd.isna(avg_lead_time):
            avg_lead_time = 7

        # =====================================================
        # 4️⃣ Annual Demand from Forecast
        # =====================================================
        forecast_data = supabase.table("demand_forecast").select("*").execute().data
        df_forecast = pd.DataFrame(forecast_data)

        if df_forecast.empty:
            log.warning("No forecast data found.")
            return

        df_forecast["product_id"] = pd.to_numeric(df_forecast["product_id"], errors="coerce")

        annual_demand = (
            df_forecast
            .groupby("product_id")["predicted_units"]
            .sum()
            .reset_index()
        )

        annual_demand["annual_demand"] = annual_demand["predicted_units"] * 52
        annual_demand = annual_demand.drop(columns=["predicted_units"])

        # =====================================================
        # 5️⃣ Merge All Data
        # =====================================================
        df = latest_features.merge(
            latest_cost[[
                "product_id",
                "ordering_cost",
                "holding_cost_per_unit",
                "stockout_cost"
            ]],
            on="product_id",
            how="left"
        ).merge(
            annual_demand,
            on="product_id",
            how="left"
        )

        df["annual_demand"] = df["annual_demand"].fillna(0)
        df["ordering_cost"] = df["ordering_cost"].fillna(1)
        df["holding_cost_per_unit"] = df["holding_cost_per_unit"].replace(0, 1)

        # =====================================================
        # 6️⃣ Safety Stock
        # =====================================================
        df["safety_stock"] = (
            service_level_z *
            df["rolling_30d_std"].fillna(0) *
            np.sqrt(avg_lead_time)
        )

        # =====================================================
        # 7️⃣ Reorder Point
        # =====================================================
        df["reorder_point"] = (
            df["rolling_30d_mean"].fillna(0) * avg_lead_time
        ) + df["safety_stock"]

        # =====================================================
        # 8️⃣ EOQ
        # =====================================================
        df["eoq"] = np.sqrt(
            (2 * df["annual_demand"] * df["ordering_cost"]) /
            df["holding_cost_per_unit"]
        )

        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)

        # =====================================================
        # 9️⃣ Cost Estimates
        # =====================================================
        df["annual_holding_cost"] = (df["eoq"] / 2) * df["holding_cost_per_unit"]

        df["annual_ordering_cost"] = (
            (df["annual_demand"] / df["eoq"].replace(0, 1)) *
            df["ordering_cost"]
        )

        df["total_inventory_cost_estimate"] = (
            df["annual_holding_cost"] +
            df["annual_ordering_cost"]
        )

        final_df = df[[
            "product_id",
            "safety_stock",
            "reorder_point",
            "eoq",
            "annual_holding_cost",
            "annual_ordering_cost",
            "total_inventory_cost_estimate"
        ]]

        # =====================================================
        # Clear old data safely
        # =====================================================
        supabase.table("inventory_optimization") \
            .delete() \
            .not_.is_("product_id", None) \
            .execute()

        # =====================================================
        # Insert new data in batches
        # =====================================================
        records = final_df.to_dict(orient="records")

        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            supabase.table("inventory_optimization").insert(batch).execute()

        log.info("Strategic Inventory Policy generated successfully.")

    except Exception as e:
        log.error(f"Strategic Inventory Optimization failed: {e}", exc_info=True)
        raise


# ------------------------------------------------------------
# AIRFLOW ENTRY
# ------------------------------------------------------------
def run(**context):

    from ingestion.config import load_settings

    log.info("Initializing Supabase for Inventory Optimization...")

    settings = load_settings(env="dev")

    supabase = create_client(
        settings.supabase_url,
        settings.supabase_key
    )

    generate_strategic_inventory_policy(supabase)


if __name__ == "__main__":
    run()