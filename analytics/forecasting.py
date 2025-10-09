import logging
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import numpy as np

def forecast_demand(supabase):
    """Forecast demand for each product using Prophet and store in Supabase."""
    logging.info("Running demand forecasting")

    # Fetch historical order data
    orders = supabase.table("fact_order_line").select(
        "order_placement_date, product_id, order_qty"
    ).execute()
    df_orders = pd.DataFrame(orders.data)

    if df_orders.empty:
        logging.warning("No orders found for forecasting")
        return []

    # Ensure date is datetime
    df_orders["order_placement_date"] = pd.to_datetime(df_orders["order_placement_date"])
    df_orders["order_qty"] = pd.to_numeric(df_orders["order_qty"], errors="coerce").fillna(0)

    forecasts = []

    for product_id, group in df_orders.groupby("product_id"):
        try:
            df = group.rename(columns={
                "order_placement_date": "ds",
                "order_qty": "y"
            }).sort_values("ds")

            # --- Data sufficiency check ---
            if len(df) < 30:
                logging.warning(f"Skipping product {product_id}: < 30 data points")
                continue

            # --- Fill missing dates ---
            df = df.set_index("ds").resample("D").sum().reset_index()
            df["y"] = df["y"].fillna(0)

            # --- Cap forecasts based on history ---
            cap = max(df["y"].max() * 1.5, 1)
            floor = 0

            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                growth="linear",
                interval_width=0.8
            )
            model.fit(df)

            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            # --- Clamp forecasts ---
            forecast["yhat"] = forecast["yhat"].clip(lower=floor, upper=cap)

            # --- Evaluate forecast (last 7 days) ---
            actual_last_7 = df.tail(7)["y"].sum()
            pred_last_7 = forecast.tail(7)["yhat"].sum()
            mape = np.abs((actual_last_7 - pred_last_7) / (actual_last_7 + 1e-6)) * 100

            if mape > 50:
                logging.warning(f"High MAPE ({mape:.1f}%) for product {product_id} — forecast may be unreliable")

            avg_demand = int(round(forecast.tail(30)["yhat"].mean()))

            supabase.table("policy_recommendations").upsert({
                "product_id": product_id,
                "forecast_demand": avg_demand,
                "forecast_accuracy": round(mape, 2)
            }).execute()

            forecasts.append({
                "product_id": product_id,
                "forecast_demand": avg_demand,
                "mape": round(mape, 2)
            })

            logging.info(
                f"Forecast for product {product_id}: {avg_demand} units (MAPE: {mape:.1f}%)"
            )

        except Exception as e:
            logging.error(f"Forecasting failed for product {product_id}: {e}")

    logging.info(f"Forecasting complete ({len(forecasts)} products)")
    return forecasts