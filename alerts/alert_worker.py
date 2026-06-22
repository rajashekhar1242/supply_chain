import os
import io
import smtplib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import imageio

from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

from supabase import create_client
from ingestion.config import load_settings
from ingestion.utils import log

plt.style.use("seaborn-v0_8")

# ==========================================================
# CONFIG
# ==========================================================
TARGET_MIN_DAYS = 7
TARGET_MAX_DAYS = 45
DAYS_COVER_CAP = 120

# If your rolling_30d_mean is actually a 30-day TOTAL, set this to False.
ROLLING_30D_MEAN_IS_DAILY_AVG = False

# Risk caps to keep health score meaningful (0..100)
MAX_INVENTORY_RISK = 60
MAX_FORECAST_RISK = 20
MAX_NETWORK_RISK = 20


# ==========================================================
# HELPERS
# ==========================================================
def _to_num(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default)

def _to_dt(s):
    return pd.to_datetime(s, errors="coerce")

def _coalesce(a, b):
    return a.where(a.notna(), b)


# ==========================================================
# HEALTH SCORE
# ==========================================================
def compute_health_score(inventory_risk, forecast_risk, network_risk):
    total_risk = float(inventory_risk) + float(forecast_risk) + float(network_risk)
    score = max(0.0, 100.0 - total_risk)

    if score >= 90:
        status = "STABLE"
    elif score >= 75:
        status = "WATCH"
    elif score >= 60:
        status = "AT RISK"
    else:
        status = "CRITICAL"

    return score, status


# ==========================================================
# DATA COLLECTION (schema-aligned)
# ==========================================================
def collect_data(supabase):
    # Inventory (product x warehouse) and policy thresholds
    mw_opt = supabase.table("multi_warehouse_optimization").select(
        "product_id, warehouse, net_stock_position, stock_status, suggested_transfer_qty, estimated_transfer_value"
    ).execute().data

    inv_policy = supabase.table("inventory_optimization").select(
        "product_id, safety_stock, reorder_point, eoq"
    ).execute().data

    # Demand features: latest per product
    demand = supabase.table("demand_features").select(
        "product_id, rolling_30d_mean, date"
    ).order("date", desc=True).limit(200000).execute().data

    # Forecast metrics: pull enough rows to include all folds of latest run
    forecast_metrics = supabase.table("demand_model_metrics").select(
        "run_timestamp, fold, mape, rmse"
    ).order("run_timestamp", desc=True).limit(5000).execute().data

    # Network plan
    network = supabase.table("network_transfer_plan").select(
        "product_id, from_warehouse, to_warehouse, allocated_qty, cost_per_unit, total_transfer_cost"
    ).execute().data

    # Warehouse dim
    dim_wh = supabase.table("dim_warehouses").select(
        "warehouse_id, warehouse_name, city, latitude, longitude"
    ).execute().data

    # Costs: include week to dedupe to latest per product
    costs = supabase.table("weekly_cost_updates").select(
        "product_id, week_start_date, unit_cost, holding_rate, stockout_cost"
    ).order("week_start_date", desc=True).limit(200000).execute().data

    return mw_opt, inv_policy, demand, forecast_metrics, network, dim_wh, costs


# ==========================================================
# METRICS
# ==========================================================
def calculate_metrics(mw_opt, inv_policy, demand, forecast_metrics, network, costs):
    df_mw = pd.DataFrame(mw_opt)
    if df_mw.empty:
        raise ValueError("No data returned from multi_warehouse_optimization (inventory base).")

    # --- normalize types
    df_mw["product_id"] = _to_num(df_mw["product_id"], default=np.nan).astype("Int64")
    df_mw["warehouse"] = _to_num(df_mw["warehouse"], default=np.nan).astype("Int64")
    df_mw["net_stock_position"] = _to_num(df_mw.get("net_stock_position"), default=0.0)

    # --- product-level stock (sum across warehouses)
    df_inv = (df_mw
              .dropna(subset=["product_id"])
              .groupby("product_id", as_index=False)
              .agg(stock_level=("net_stock_position", "sum")))

    # --- demand: latest date per product
    df_dem = pd.DataFrame(demand)
    if df_dem.empty:
        df_dem = pd.DataFrame(columns=["product_id", "rolling_30d_mean", "date"])

    df_dem["product_id"] = _to_num(df_dem["product_id"], default=np.nan).astype("Int64")
    df_dem["rolling_30d_mean"] = _to_num(df_dem.get("rolling_30d_mean"), default=0.0)
    df_dem["date"] = _to_dt(df_dem.get("date"))

    df_dem = (df_dem
              .dropna(subset=["product_id"])
              .sort_values(["product_id", "date"])
              .drop_duplicates(subset=["product_id"], keep="last"))

    df = df_inv.merge(df_dem[["product_id", "rolling_30d_mean"]], on="product_id", how="left")
    df["rolling_30d_mean"] = df["rolling_30d_mean"].fillna(0.0)

    # Interpret demand
    if ROLLING_30D_MEAN_IS_DAILY_AVG:
        df["daily_demand"] = df["rolling_30d_mean"]
    else:
        df["daily_demand"] = df["rolling_30d_mean"] / 30.0

    # --- costs: latest week per product
    df_cost = pd.DataFrame(costs)
    if df_cost.empty:
        df_cost = pd.DataFrame(columns=["product_id", "week_start_date", "unit_cost", "holding_rate", "stockout_cost"])

    df_cost["product_id"] = _to_num(df_cost["product_id"], default=np.nan).astype("Int64")
    df_cost["week_start_date"] = _to_dt(df_cost.get("week_start_date"))
    df_cost["unit_cost"] = _to_num(df_cost.get("unit_cost"), default=np.nan)
    df_cost["holding_rate"] = _to_num(df_cost.get("holding_rate"), default=np.nan)
    df_cost["stockout_cost"] = _to_num(df_cost.get("stockout_cost"), default=np.nan)

    df_cost = (df_cost
               .dropna(subset=["product_id"])
               .sort_values(["product_id", "week_start_date"])
               .drop_duplicates(subset=["product_id"], keep="last"))

    df = df.merge(df_cost[["product_id", "unit_cost", "holding_rate", "stockout_cost"]], on="product_id", how="left")

    df["unit_cost_missing"] = df["unit_cost"].isna()
    df["unit_cost"] = df["unit_cost"].fillna(0.0)

    # --- inventory policy thresholds
    df_pol = pd.DataFrame(inv_policy)
    if df_pol.empty:
        df_pol = pd.DataFrame(columns=["product_id", "safety_stock", "reorder_point", "eoq"])

    df_pol["product_id"] = _to_num(df_pol.get("product_id"), default=np.nan).astype("Int64")
    df_pol["safety_stock"] = _to_num(df_pol.get("safety_stock"), default=np.nan)
    df_pol["reorder_point"] = _to_num(df_pol.get("reorder_point"), default=np.nan)
    df_pol["eoq"] = _to_num(df_pol.get("eoq"), default=np.nan)

    df_pol = df_pol.dropna(subset=["product_id"]).drop_duplicates(subset=["product_id"], keep="last")
    df = df.merge(df_pol[["product_id", "safety_stock", "reorder_point", "eoq"]], on="product_id", how="left")

    # --- days cover
    df["days_cover"] = np.where(df["daily_demand"] > 0, df["stock_level"] / df["daily_demand"], np.nan)
    df["days_cover"] = df["days_cover"].clip(upper=DAYS_COVER_CAP)

    # --- classification: policy-based when available; fallback to days-cover bands
    # Policy targets
    # min_stock: reorder_point if available else daily_demand * TARGET_MIN_DAYS
    # max_stock: (eoq + safety_stock) if available else daily_demand * TARGET_MAX_DAYS
    df["min_stock"] = _coalesce(df["reorder_point"], df["daily_demand"] * TARGET_MIN_DAYS)
    df["max_stock"] = _coalesce(df["eoq"] + df["safety_stock"], df["daily_demand"] * TARGET_MAX_DAYS)

    df["shortage_units"] = np.maximum(0.0, df["min_stock"] - df["stock_level"])
    df["excess_units"] = np.maximum(0.0, df["stock_level"] - df["max_stock"])

    df["inventory_bucket"] = np.select(
        [
            df["shortage_units"] > 0,
            df["excess_units"] > 0
        ],
        [
            "STOCKOUT_RISK",
            "OVERSTOCK"
        ],
        default="OPTIMAL"
    )

    low_stock = df[df["inventory_bucket"] == "STOCKOUT_RISK"]
    overstock = df[df["inventory_bucket"] == "OVERSTOCK"]

    low_stock_count = int(len(low_stock))
    overstock_count = int(len(overstock))

    # --- forecast risk: latest run across folds
    df_f = pd.DataFrame(forecast_metrics)
    if not df_f.empty:
        df_f["run_timestamp"] = _to_dt(df_f.get("run_timestamp"))
        latest_ts = df_f["run_timestamp"].max()
        latest_run = df_f[df_f["run_timestamp"] == latest_ts]
        forecast_mape = float(_to_num(latest_run.get("mape"), default=np.nan).dropna().mean() or 0.0)
    else:
        forecast_mape = 0.0

    # Forecast risk (bounded)
    if forecast_mape >= 30:
        forecast_risk = 20
    elif forecast_mape >= 20:
        forecast_risk = 12
    elif forecast_mape >= 15:
        forecast_risk = 8
    else:
        forecast_risk = 0
    forecast_risk = min(MAX_FORECAST_RISK, forecast_risk)

    # --- network cost + risk
    df_net = pd.DataFrame(network)
    network_cost = 0.0
    network_risk = 0.0

    if not df_net.empty:
        df_net["allocated_qty"] = _to_num(df_net.get("allocated_qty"), default=0.0)
        df_net["cost_per_unit"] = _to_num(df_net.get("cost_per_unit"), default=0.0)
        df_net["total_transfer_cost"] = pd.to_numeric(df_net.get("total_transfer_cost"), errors="coerce")

        df_net["calc_cost"] = df_net["allocated_qty"] * df_net["cost_per_unit"]
        df_net["effective_cost"] = _coalesce(df_net["total_transfer_cost"], df_net["calc_cost"]).fillna(0.0)

        network_cost = float(df_net["effective_cost"].sum())

        # High-cost route count (use percentile threshold so it scales with your data)
        if len(df_net) >= 10:
            p90 = float(df_net["effective_cost"].quantile(0.90))
            high_cost_routes = int((df_net["effective_cost"] >= p90).sum())
        else:
            high_cost_routes = int((df_net["effective_cost"] > 100000).sum())

        network_risk = min(MAX_NETWORK_RISK, high_cost_routes * 2)

    # --- financial exposure (risk-based, not total value)
    # Stockout exposure: shortage_units * stockout_cost (fallback to unit_cost)
    df["effective_stockout_cost"] = _coalesce(df["stockout_cost"], df["unit_cost"]).fillna(0.0)
    stockout_exposure = float((df["shortage_units"] * df["effective_stockout_cost"]).sum())

    # Overstock exposure: excess units * unit_cost (capital tied up)
    overstock_exposure = float((df["excess_units"] * df["unit_cost"]).sum())

    # Optional: holding cost overlay if holding_rate present (annual)
    df["holding_rate"] = df["holding_rate"].fillna(0.0)
    holding_cost_exposure = float((df["excess_units"] * df["unit_cost"] * df["holding_rate"]).sum())

    total_exposure = stockout_exposure + overstock_exposure + network_cost

    # --- inventory risk (bounded, severity-aware)
    # Severity = shortage_units / (min_stock + 1) aggregated
    df["shortage_severity"] = np.where(df["min_stock"] > 0, df["shortage_units"] / (df["min_stock"] + 1e-6), 0.0)
    inv_risk_raw = float((df["shortage_severity"].clip(0, 1)).sum() * 5)  # scale factor
    inventory_risk = min(MAX_INVENTORY_RISK, inv_risk_raw)

    return {
        "low_stock_count": low_stock_count,
        "overstock_count": overstock_count,

        "forecast_mape": float(forecast_mape),

        "stockout_exposure": float(stockout_exposure),
        "overstock_exposure": float(overstock_exposure),
        "holding_cost_exposure": float(holding_cost_exposure),
        "network_cost": float(network_cost),
        "total_exposure": float(total_exposure),

        "inventory_risk": float(inventory_risk),
        "forecast_risk": float(forecast_risk),
        "network_risk": float(network_risk),

        # Useful diagnostics
        "products_missing_unit_cost": int(df["unit_cost_missing"].sum()),
        "total_products": int(df["product_id"].nunique()),
    }


# ==========================================================
# VISUALS
# ==========================================================
def generate_health_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Supply Chain Health Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 60], "color": "#C62828"},
                {"range": [60, 75], "color": "#EF6C00"},
                {"range": [75, 90], "color": "#F9A825"},
                {"range": [90, 100], "color": "#2E7D32"},
            ],
        }
    ))
    return fig.to_image(format="png")


def generate_inventory_chart(metrics):
    fig, ax = plt.subplots(figsize=(6, 4))
    values = [metrics["low_stock_count"], metrics["overstock_count"]]
    bars = ax.bar(["Stockout Risk", "Overstock"], values)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h, f"{int(h)}", ha="center", va="bottom")
    ax.set_title("Inventory Risk Distribution")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_exposure_chart(metrics):
    values = [metrics["stockout_exposure"], metrics["overstock_exposure"], metrics["network_cost"]]
    values = [v if v > 0 else 0.01 for v in values]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(values, labels=["Stockout Risk", "Overstock", "Network"], autopct="%1.1f%%", startangle=140)
    ax.set_title("Financial Exposure Breakdown")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_geo_map(dim_wh, mw_opt, network):
    df_dim = pd.DataFrame(dim_wh)
    df_status = pd.DataFrame(mw_opt)
    df_net = pd.DataFrame(network)

    if df_dim.empty or df_status.empty:
        return None

    df_dim["warehouse_id"] = _to_num(df_dim["warehouse_id"], default=np.nan).astype("Int64")
    df_status["warehouse"] = _to_num(df_status["warehouse"], default=np.nan).astype("Int64")

    # Merge status + dim
    df = df_status.merge(df_dim, left_on="warehouse", right_on="warehouse_id", how="left")

    # Reduce to one row per warehouse: worst severity wins
    df["severity"] = np.where(
        df["stock_status"] == "Shortage", "CRITICAL",
        np.where(df["stock_status"] == "Surplus", "WATCH", "STABLE")
    )
    sev_rank = {"STABLE": 1, "WATCH": 2, "CRITICAL": 3}
    df["sev_rank"] = df["severity"].map(sev_rank).fillna(1)
    df = df.sort_values("sev_rank").drop_duplicates(subset=["warehouse_id"], keep="last")

    fig = go.Figure()

    critical = df[df["severity"] == "CRITICAL"]
    fig.add_trace(go.Scattergeo(
        lon=critical["longitude"], lat=critical["latitude"],
        mode="markers",
        marker=dict(size=35, color="red", opacity=0.25),
        showlegend=False
    ))

    color_map = {"CRITICAL": "red", "WATCH": "orange", "STABLE": "green"}
    fig.add_trace(go.Scattergeo(
        lon=df["longitude"], lat=df["latitude"],
        text=df["warehouse_name"],
        mode="markers+text",
        marker=dict(size=14, color=df["severity"].map(color_map)),
        name="Warehouses"
    ))

    # Routes: aggregate by (from,to) and use allocated_qty (schema-correct)
    if not df_net.empty:
        df_net["from_warehouse"] = _to_num(df_net["from_warehouse"], default=np.nan).astype("Int64")
        df_net["to_warehouse"] = _to_num(df_net["to_warehouse"], default=np.nan).astype("Int64")
        df_net["allocated_qty"] = _to_num(df_net.get("allocated_qty"), default=0.0)

        routes = (df_net.dropna(subset=["from_warehouse", "to_warehouse"])
                        .groupby(["from_warehouse", "to_warehouse"], as_index=False)
                        .agg(allocated_qty=("allocated_qty", "sum")))

        routes = routes.merge(df_dim, left_on="from_warehouse", right_on="warehouse_id", how="left") \
                       .rename(columns={"latitude": "from_lat", "longitude": "from_lon"})
        routes = routes.merge(df_dim, left_on="to_warehouse", right_on="warehouse_id", how="left") \
                       .rename(columns={"latitude": "to_lat", "longitude": "to_lon"})

        for _, r in routes.iterrows():
            qty = float(r.get("allocated_qty", 0.0))
            width = max(1, min(qty / 500.0, 6))

            fig.add_trace(go.Scattergeo(
                lon=[r["from_lon"], r["to_lon"]],
                lat=[r["from_lat"], r["to_lat"]],
                mode="lines",
                line=dict(width=width, color="blue"),
                opacity=0.6,
                showlegend=False
            ))
            fig.add_trace(go.Scattergeo(
                lon=[r["to_lon"]], lat=[r["to_lat"]],
                mode="markers",
                marker=dict(size=8, color="blue", symbol="triangle-up"),
                showlegend=False
            ))

    fig.update_layout(
        geo=dict(
            scope="asia",
            center=dict(lat=22, lon=80),
            projection_scale=6,
            showland=True,
            showcountries=True,
            countrycolor="gray",
            landcolor="rgb(243,243,243)"
        ),
        title="India Warehouse Transfer Network",
        height=600
    )

    return fig.to_image(format="png")


def generate_flow_animation(dim_wh, network):
    df_dim = pd.DataFrame(dim_wh)
    df_net = pd.DataFrame(network)

    if df_dim.empty or df_net.empty:
        return None

    df_dim["warehouse_id"] = _to_num(df_dim["warehouse_id"], default=np.nan).astype("Int64")
    df_net["from_warehouse"] = _to_num(df_net["from_warehouse"], default=np.nan).astype("Int64")
    df_net["to_warehouse"] = _to_num(df_net["to_warehouse"], default=np.nan).astype("Int64")
    df_net["allocated_qty"] = _to_num(df_net.get("allocated_qty"), default=0.0)

    routes = (df_net.dropna(subset=["from_warehouse", "to_warehouse"])
                    .groupby(["from_warehouse", "to_warehouse"], as_index=False)
                    .agg(allocated_qty=("allocated_qty", "sum")))

    routes = routes.merge(df_dim, left_on="from_warehouse", right_on="warehouse_id", how="left") \
                   .rename(columns={"latitude": "from_lat", "longitude": "from_lon"})
    routes = routes.merge(df_dim, left_on="to_warehouse", right_on="warehouse_id", how="left") \
                   .rename(columns={"latitude": "to_lat", "longitude": "to_lon"})

    frames = []

    for step in np.linspace(0, 1, 12):
        fig = go.Figure()

        fig.add_trace(go.Scattergeo(
            lon=df_dim["longitude"], lat=df_dim["latitude"],
            mode="markers",
            marker=dict(size=12, color="green")
        ))

        for _, r in routes.iterrows():
            lon = r["from_lon"] + (r["to_lon"] - r["from_lon"]) * step
            lat = r["from_lat"] + (r["to_lat"] - r["from_lat"]) * step
            fig.add_trace(go.Scattergeo(
                lon=[lon], lat=[lat],
                mode="markers",
                marker=dict(size=8, color="blue"),
                showlegend=False
            ))

        fig.update_layout(
            geo=dict(scope="asia", center=dict(lat=22, lon=80), projection_scale=4.5),
            showlegend=False
        )

        img = fig.to_image(format="png")
        frames.append(imageio.v2.imread(io.BytesIO(img)))

    gif_buffer = io.BytesIO()
    imageio.mimsave(gif_buffer, frames, format="GIF", duration=0.25)
    gif_buffer.seek(0)
    return gif_buffer.read()


# ==========================================================
# EXECUTIVE HTML TEMPLATE
# ==========================================================
def build_executive_html(metrics, health_score, status):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    status_color = {
        "STABLE": "#2E7D32",
        "WATCH": "#F9A825",
        "AT RISK": "#EF6C00",
        "CRITICAL": "#C62828"
    }[status]

    insights = []

    if metrics["low_stock_count"] > 5:
        insights.append("Elevated stockout risk across multiple SKUs.")
    if metrics["overstock_count"] > 5:
        insights.append("Excess inventory impacting working capital.")
    if metrics["forecast_mape"] > 15:
        insights.append("Forecast accuracy degradation detected.")
    if metrics["network_cost"] > 200000:
        insights.append("Transfer network cost above optimal range.")
    if metrics.get("products_missing_unit_cost", 0) > 0:
        insights.append(f"{metrics['products_missing_unit_cost']} SKUs missing unit cost; exposure may be understated.")

    if not insights:
        insights.append("Operations remain stable with no major risk indicators.")

    insight_text = " ".join(insights)

    return f"""
    <div style="font-family:Segoe UI;background:#F4F6F9;padding:25px;">

        <div style="background:#1F4E79;color:white;padding:18px;border-radius:8px;">
            <span style="font-size:22px;font-weight:bold;">Supply Chain Control Tower</span>
            <div style="float:right;">{now}</div>
        </div>

        <div style="margin-top:20px;padding:15px;border-radius:8px;background:{status_color};color:white;">
            <h2 style="margin:0;">System Status: {status}</h2>
            <p style="margin:5px 0 0 0;">Health Score: {health_score:.1f} / 100</p>
        </div>

        <div style="display:flex;gap:20px;margin-top:20px;">
            <div style="background:white;padding:20px;border-radius:8px;flex:1;">
                <h4>Total Financial Exposure</h4>
                <h2>₹ {metrics['total_exposure']:,.0f}</h2>
            </div>

            <div style="background:white;padding:20px;border-radius:8px;flex:1;">
                <h4>Forecast MAPE</h4>
                <h2>{metrics['forecast_mape']:.2f}%</h2>
            </div>

            <div style="background:white;padding:20px;border-radius:8px;flex:1;">
                <h4>Network Cost</h4>
                <h2>₹ {metrics['network_cost']:,.0f}</h2>
            </div>
        </div>

        <div style="margin-top:25px;background:white;padding:20px;border-radius:8px;">
            <h3>Executive Insights</h3>
            <p style="font-size:14px;color:#444;">{insight_text}</p>
        </div>

        <div style="margin-top:25px;">
            <h3>Supply Chain Health</h3>
            <img src="cid:image1">
        </div>

        <div style="margin-top:25px;">
            <h3>Inventory Risk Overview</h3>
            <img src="cid:image2">
        </div>

        <div style="margin-top:25px;">
            <h3>Financial Exposure Breakdown</h3>
            <img src="cid:image3">
        </div>

        <div style="margin-top:25px;">
            <h3>Warehouse Optimization Map</h3>
            <img src="cid:image4">
            <div style="margin-top:25px;">
                <h3>Live Transfer Flow</h3>
                <img src="cid:image5">
            </div>
        </div>

        <div style="margin-top:30px;font-size:12px;color:#777;">
            Automated Enterprise Control Tower Report
        </div>
    </div>
    """


# ==========================================================
# EMAIL
# ==========================================================
def send_email(html_body, status, attachments):
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASS = os.getenv("SMTP_PASS")
    ALERT_TO = os.getenv("ALERT_TO")

    subject = f"Supply Chain Control Tower | {status}"

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = ALERT_TO

    msg.attach(MIMEText(html_body, "html"))

    for idx, img_bytes in enumerate(attachments, start=1):
        if img_bytes is None:
            continue
        img = MIMEImage(img_bytes)
        img.add_header("Content-ID", f"<image{idx}>")
        msg.attach(img)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, ALERT_TO.split(","), msg.as_string())


# ==========================================================
# OPTIONAL: Persist snapshot (schema table exists)
# ==========================================================
def write_daily_snapshot(supabase, health_score):
    # control_tower_daily_snapshot(snapshot_time, health_score)
    payload = {
        "snapshot_time": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds"),
        "health_score": float(health_score),
    }
    supabase.table("control_tower_daily_snapshot").insert(payload).execute()


# ==========================================================
# MAIN
# ==========================================================
def run():
    settings = load_settings(env="dev")
    supabase = create_client(settings.supabase_url, settings.supabase_key)

    mw_opt, inv_policy, demand, forecast_metrics, network, dim_wh, costs = collect_data(supabase)

    metrics = calculate_metrics(mw_opt, inv_policy, demand, forecast_metrics, network, costs)

    health_score, status = compute_health_score(
        metrics["inventory_risk"],
        metrics["forecast_risk"],
        metrics["network_risk"]
    )

    inventory_chart = generate_inventory_chart(metrics)
    exposure_chart = generate_exposure_chart(metrics)
    geo_map = generate_geo_map(dim_wh, mw_opt, network)
    flow_gif = generate_flow_animation(dim_wh, network)
    health_gauge = generate_health_gauge(health_score)

    html_body = build_executive_html(metrics, health_score, status)

    send_email(
        html_body,
        status,
        attachments=[health_gauge, inventory_chart, exposure_chart, geo_map, flow_gif]
    )

    # Optional persistence for dashboard trending
    try:
        write_daily_snapshot(supabase, health_score)
    except Exception as e:
        log.warning(f"Snapshot write failed: {e}")

    log.info("Board-level executive report sent.")
    return {"health_score": health_score, "status": status, "metrics": metrics}


if __name__ == "__main__":
    run()