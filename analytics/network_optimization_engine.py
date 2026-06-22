import pandas as pd
import numpy as np
from ingestion.utils import log
from supabase import create_client


# ------------------------------------------------------------
# CORE NETWORK OPTIMIZATION ENGINE (SUPABASE VERSION)
# ------------------------------------------------------------
def generate_cost_minimized_transfer_plan(supabase):

    try:
        log.info("Starting Cost-Minimized Network Optimization...")

        # -------------------------------------------------
        # 1️⃣ Load weekly_transfer_updates (Pagination)
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
            log.warning("No transfer data available.")
            return

        df_transfer = pd.DataFrame(all_data)

        df_transfer["transfer_qty"] = pd.to_numeric(
            df_transfer["transfer_qty"], errors="coerce"
        ).fillna(0)

        df_transfer["transfer_cost"] = pd.to_numeric(
            df_transfer["transfer_cost"], errors="coerce"
        ).fillna(0)

        # -------------------------------------------------
        # 2️⃣ Compute Net Stock Position
        # -------------------------------------------------
        outgoing = (
            df_transfer.groupby(["product_id", "from_warehouse"])["transfer_qty"]
            .sum()
            .reset_index()
            .rename(columns={
                "from_warehouse": "warehouse",
                "transfer_qty": "outgoing_qty"
            })
        )

        incoming = (
            df_transfer.groupby(["product_id", "to_warehouse"])["transfer_qty"]
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
        # 3️⃣ Compute Average Route Cost
        # -------------------------------------------------
        route_cost = (
            df_transfer.groupby(
                ["product_id", "from_warehouse", "to_warehouse"]
            )
            .agg({
                "transfer_cost": "sum",
                "transfer_qty": "sum"
            })
            .reset_index()
        )

        route_cost["cost_per_unit"] = np.where(
            route_cost["transfer_qty"] == 0,
            0,
            route_cost["transfer_cost"] / route_cost["transfer_qty"]
        )

        route_cost.replace([np.inf, -np.inf], 0, inplace=True)
        route_cost.fillna(0, inplace=True)

        # -------------------------------------------------
        # 4️⃣ Greedy Allocation Logic
        # -------------------------------------------------
        allocation_rows = []

        for product in df_net["product_id"].unique():

            product_net = df_net[df_net["product_id"] == product].copy()

            surplus = product_net[
                product_net["net_stock_position"] > 0
            ][["warehouse", "net_stock_position"]]

            shortage = product_net[
                product_net["net_stock_position"] < 0
            ][["warehouse", "net_stock_position"]]

            if surplus.empty or shortage.empty:
                continue

            surplus_dict = {
                row["warehouse"]: row["net_stock_position"]
                for _, row in surplus.iterrows()
            }

            shortage_dict = {
                row["warehouse"]: abs(row["net_stock_position"])
                for _, row in shortage.iterrows()
            }

            product_routes = route_cost[
                route_cost["product_id"] == product
            ].sort_values("cost_per_unit")

            for _, route in product_routes.iterrows():

                from_wh = route["from_warehouse"]
                to_wh = route["to_warehouse"]
                cost = route["cost_per_unit"]

                if from_wh in surplus_dict and to_wh in shortage_dict:

                    available = surplus_dict[from_wh]
                    needed = shortage_dict[to_wh]

                    transfer_qty = min(available, needed)

                    if transfer_qty > 0:

                        allocation_rows.append({
                            "product_id": int(product),
                            "from_warehouse": int(from_wh),
                            "to_warehouse": int(to_wh),
                            "allocated_qty": float(transfer_qty),
                            "cost_per_unit": float(cost),
                            "total_transfer_cost": float(transfer_qty * cost)
                        })

                        surplus_dict[from_wh] -= transfer_qty
                        shortage_dict[to_wh] -= transfer_qty

                        if surplus_dict[from_wh] == 0:
                            del surplus_dict[from_wh]

                        if shortage_dict[to_wh] == 0:
                            del shortage_dict[to_wh]

                if not surplus_dict or not shortage_dict:
                    break

        df_alloc = pd.DataFrame(allocation_rows)

        # -------------------------------------------------
        # Safe Delete (Supabase requires WHERE)
        # -------------------------------------------------
        supabase.table("network_transfer_plan") \
            .delete() \
            .not_.is_("product_id", None) \
            .execute()

        # -------------------------------------------------
        # Insert Results in Batches
        # -------------------------------------------------
        if not df_alloc.empty:

            records = df_alloc.to_dict(orient="records")

            batch_size = 500
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                supabase.table("network_transfer_plan").insert(batch).execute()

        log.info("Cost-Minimized Transfer Plan generated successfully.")

    except Exception as e:
        log.error(f"Network Optimization failed: {e}", exc_info=True)
        raise


# ------------------------------------------------------------
# AIRFLOW ENTRY
# ------------------------------------------------------------
def run(**context):

    from ingestion.config import load_settings

    log.info("Initializing Supabase for Network Optimization...")

    settings = load_settings(env="dev")

    supabase = create_client(
        settings.supabase_url,
        settings.supabase_key
    )

    generate_cost_minimized_transfer_plan(supabase)


if __name__ == "__main__":
    run()