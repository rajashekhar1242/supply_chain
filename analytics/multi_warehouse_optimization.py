import pandas as pd
import numpy as np
from ingestion.utils import log
from supabase import create_client


# ------------------------------------------------------------
# CORE MULTI-WAREHOUSE ENGINE (SUPABASE VERSION)
# ------------------------------------------------------------
def generate_multi_warehouse_optimization(supabase):

    try:
        log.info("Starting Multi-Warehouse Optimization...")

        # -------------------------------------------------
        # Fetch weekly_transfer_updates (Pagination)
        # -------------------------------------------------
        all_data = []
        page_size = 1000
        start = 0

        while True:
            response = (
                supabase
                .table("weekly_transfer_updates")
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
            log.warning("No transfer data found.")
            return

        df_transfer = pd.DataFrame(all_data)

        df_transfer["transfer_qty"] = pd.to_numeric(
            df_transfer["transfer_qty"], errors="coerce"
        ).fillna(0)

        df_transfer["transfer_cost"] = pd.to_numeric(
            df_transfer["transfer_cost"], errors="coerce"
        ).fillna(0)

        # -------------------------------------------------
        # Compute Net Flow per Warehouse
        # -------------------------------------------------
        outgoing = (
            df_transfer.groupby(
                ["product_id", "from_warehouse"]
            )["transfer_qty"]
            .sum()
            .reset_index()
            .rename(columns={
                "from_warehouse": "warehouse",
                "transfer_qty": "outgoing_qty"
            })
        )

        incoming = (
            df_transfer.groupby(
                ["product_id", "to_warehouse"]
            )["transfer_qty"]
            .sum()
            .reset_index()
            .rename(columns={
                "to_warehouse": "warehouse",
                "transfer_qty": "incoming_qty"
            })
        )

        df_net = outgoing.merge(
            incoming,
            on=["product_id", "warehouse"],
            how="outer"
        ).fillna(0)

        df_net["net_stock_position"] = (
            df_net["incoming_qty"] - df_net["outgoing_qty"]
        )

        # -------------------------------------------------
        # Identify Surplus & Shortage
        # -------------------------------------------------
        df_net["stock_status"] = np.where(
            df_net["net_stock_position"] > 0,
            "Surplus",
            np.where(
                df_net["net_stock_position"] < 0,
                "Shortage",
                "Balanced"
            )
        )

        # -------------------------------------------------
        # Average Transfer Cost Per Unit
        # -------------------------------------------------
        cost_group = (
            df_transfer.groupby("product_id")
            .agg({
                "transfer_cost": "sum",
                "transfer_qty": "sum"
            })
            .reset_index()
        )

        cost_group["avg_transfer_cost_per_unit"] = np.where(
            cost_group["transfer_qty"] == 0,
            0,
            cost_group["transfer_cost"] / cost_group["transfer_qty"]
        )

        df_net = df_net.merge(
            cost_group[["product_id", "avg_transfer_cost_per_unit"]],
            on="product_id",
            how="left"
        )

        # -------------------------------------------------
        # Suggested Transfer Qty (50% of surplus)
        # -------------------------------------------------
        df_net["suggested_transfer_qty"] = np.where(
            df_net["stock_status"] == "Surplus",
            df_net["net_stock_position"] * 0.5,
            0
        )

        # -------------------------------------------------
        # Estimated Transfer Value
        # -------------------------------------------------
        df_net["estimated_transfer_value"] = (
            df_net["suggested_transfer_qty"] *
            df_net["avg_transfer_cost_per_unit"]
        )

        # Clean NaN / inf
        df_net.replace([np.inf, -np.inf], 0, inplace=True)
        df_net.fillna(0, inplace=True)

        final_df = df_net[[
            "product_id",
            "warehouse",
            "net_stock_position",
            "stock_status",
            "suggested_transfer_qty",
            "estimated_transfer_value"
        ]]

        # -------------------------------------------------
        # Safe Delete (Supabase requires WHERE)
        # -------------------------------------------------
        supabase.table("multi_warehouse_optimization") \
            .delete() \
            .not_.is_("product_id", None) \
            .execute()

        # -------------------------------------------------
        # Insert in batches
        # -------------------------------------------------
        records = final_df.to_dict(orient="records")

        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            supabase.table("multi_warehouse_optimization").insert(batch).execute()

        log.info("Multi-Warehouse Optimization completed successfully.")

    except Exception as e:
        log.error(f"Multi-Warehouse Optimization failed: {e}", exc_info=True)
        raise


# ------------------------------------------------------------
# AIRFLOW ENTRY
# ------------------------------------------------------------
def run(**context):

    from ingestion.config import load_settings

    log.info("Initializing Supabase for Multi-Warehouse Optimization...")

    settings = load_settings(env="dev")

    supabase = create_client(
        settings.supabase_url,
        settings.supabase_key
    )

    generate_multi_warehouse_optimization(supabase)


if __name__ == "__main__":
    run()