import os
import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

load_dotenv()

st.set_page_config(page_title="Supply Chain Control Tower", layout="wide")
st.title("Supply Chain Control Tower")

# -----------------------------------------------------
# DATABASE CONNECTION
# -----------------------------------------------------

@st.cache_resource
def get_engine():

    engine = create_engine(
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT','5432')}/{os.getenv('DB_NAME')}",
        pool_pre_ping=True,
        pool_recycle=1800
    )

    return engine

engine = get_engine()

# -----------------------------------------------------
# QUERY HELPER
# -----------------------------------------------------

def run_query(sql, params=None):

    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn, params=params)

    return df

def safe_query(sql: str, params=None):
    params = params or {}
    try:
        with engine.connect() as conn:
            return pd.read_sql_query(text(sql), conn, params=params)
    except Exception as exc:
        st.error(f"⚠️ Query failed: {exc}")
        return pd.DataFrame()


def yes_no_rate(expr: str, total_expr="COUNT(*)"):
    """Helper for computing YES/NO ratios safely."""
    return f"ROUND(COALESCE({expr}, 0) * 100.0 / NULLIF({total_expr}, 0), 2)"

# -----------------------------------------------------
# DATE RANGE
# -----------------------------------------------------

date_sql = """
SELECT MIN(actual_delivery_date) as min_date,
MAX(actual_delivery_date) as max_date
FROM fact_order_line
"""

date_df = run_query(date_sql)

default_start = date_df["min_date"][0]
default_end = date_df["max_date"][0]

with st.sidebar:

    st.header("Filters")

    start_date = st.date_input("Start Date", default_start)
    end_date = st.date_input("End Date", default_end)

# -----------------------------------------------------
# KPI METRICS
# -----------------------------------------------------

kpi_sql = """
SELECT
COUNT(*) as total_orders,
SUM(CASE WHEN on_time='1' THEN 1 ELSE 0 END)*100.0/COUNT(*) as on_time_rate,
SUM(CASE WHEN in_full='1' THEN 1 ELSE 0 END)*100.0/COUNT(*) as in_full_rate,
SUM(CASE WHEN on_time_in_full='1' THEN 1 ELSE 0 END)*100.0/COUNT(*) as otif_rate
FROM fact_order_line
WHERE actual_delivery_date BETWEEN :start_date AND :end_date
"""

params = {"start_date": start_date, "end_date": end_date}

kpi_df = run_query(kpi_sql, params)

total_orders = int(kpi_df["total_orders"][0])
otif = round(kpi_df["otif_rate"][0],2)
on_time = round(kpi_df["on_time_rate"][0],2)
in_full = round(kpi_df["in_full_rate"][0],2)

health_score = round((otif + on_time + in_full)/3,2)

k1,k2,k3,k4,k5 = st.columns(5)

k1.metric("Orders", total_orders)
k2.metric("OTIF %", otif)
k3.metric("On Time %", on_time)
k4.metric("In Full %", in_full)
k5.metric("Health Score", health_score)

st.divider()

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=health_score,
    title={"text":"Supply Chain Health"},
    gauge={
        "axis":{"range":[0,100]},
        "steps":[
            {"range":[0,50],"color":"red"},
            {"range":[50,75],"color":"orange"},
            {"range":[75,100],"color":"green"}
        ]
    }
))

st.plotly_chart(fig, use_container_width=True, key="health_score")

# -----------------------------------------------------
# PRODUCT LIST
# -----------------------------------------------------

products_df = run_query("""
SELECT DISTINCT product_id
FROM demand_forecast
ORDER BY product_id
""")

product_list = products_df["product_id"].astype(str).tolist()

# -----------------------------------------------------
# TABS
# -----------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Demand Dashboard",
    "Inventory Dashboard",
    "Supplier Dashboard",
    "Performance Dashboard",
    "Logistics & Warehouse Dashboard",
    "Control Tower Alerts"
])

# -----------------------------------------------------
# DEMAND INTELLIGENCE
# -----------------------------------------------------

with tab1:

    st.subheader("Demand Intelligence")

    col1, col2 = st.columns([1,4])

    with col1:
        selected_product = st.selectbox("Select Product", product_list)

    # -------------------------------
    # FORECAST DATA
    # -------------------------------
    with st.spinner("Loading forecast..."):
        forecast_df = run_query("""
        SELECT date, predicted_units
        FROM demand_forecast
        WHERE product_id=:product
        ORDER BY date
        """, {"product": selected_product})

    if forecast_df.empty:
        st.warning("No forecast data available")
    else:
        # -------------------------------
        # DATA PREPROCESSING
        # -------------------------------
        forecast_df["date"] = pd.to_datetime(forecast_df["date"])
        forecast_df = forecast_df.sort_values("date")

        # ✅ FIX 1: Stable moving average
        forecast_df["ma_7"] = forecast_df["predicted_units"].rolling(7, min_periods=1).mean()

        # ✅ FIX 2: Break line for date gaps (IMPORTANT)
        forecast_df["gap"] = forecast_df["date"].diff().dt.days
        forecast_df.loc[forecast_df["gap"] > 2, "predicted_units"] = None

        with col2:
            fig = go.Figure()

            # Forecast line
            fig.add_trace(go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["predicted_units"],
                mode="lines+markers",
                name="Forecast"
            ))

            # Trend line
            fig.add_trace(go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["ma_7"],
                mode="lines",
                name="7-Day Trend",
                line=dict(dash="dot")
            ))

            fig.update_layout(
                title=f"Demand Forecast + Trend ({selected_product})",
                template="plotly_dark"
            )

            fig.update_xaxes(tickformat="%b %d")

            st.plotly_chart(fig, use_container_width=True)

        # -------------------------------
        # BUSINESS INSIGHT (NEW 🔥)
        # -------------------------------
        latest = forecast_df["predicted_units"].dropna().iloc[-1]
        avg = forecast_df["predicted_units"].mean()

        if latest > avg:
            st.info("Demand is trending upward")
        else:
            st.info("Demand is stable or decreasing")

    # -------------------------------
    # VOLATILITY
    # -------------------------------
    st.subheader("Product Demand Volatility")

    vol_product = run_query("""
    SELECT AVG(volatility_index) as volatility
    FROM demand_features
    WHERE product_id=:product
    """, {"product": selected_product})

    if not vol_product.empty and vol_product["volatility"][0] is not None:
        st.metric("Volatility Index", round(vol_product["volatility"][0], 2))
    else:
        st.warning("No volatility data available")

    # -------------------------------
    # INVENTORY POLICY
    # -------------------------------
    st.subheader("Inventory Policy")

    inventory_product = run_query("""
    SELECT safety_stock, reorder_point, eoq
    FROM inventory_optimization
    WHERE product_id=:product
    """, {"product": selected_product})

    if not inventory_product.empty:
        st.dataframe(inventory_product)
    else:
        st.warning("No inventory policy data available")





with tab2:

    st.subheader("Inventory & Replenishment Intelligence")

    with st.spinner("Loading inventory data..."):
        ml_df = run_query("""
        SELECT d.product_id,
        d.rolling_7d_mean,
        d.rolling_30d_mean,
        d.volatility_index,
        i.safety_stock,
        i.reorder_point,
        i.eoq,
        i.total_inventory_cost_estimate
        FROM demand_features d
        JOIN inventory_optimization i
        ON d.product_id=i.product_id
        WHERE d.product_id=:product
        """, {"product": selected_product})

    if ml_df.empty:
        st.warning("No inventory data available")
    else:
        # -------------------------------
        # 🧠 DATA CLEANING
        # -------------------------------
        ml_df = ml_df.replace([np.inf, -np.inf], np.nan)

        ml_df["rolling_7d_mean"] = ml_df["rolling_7d_mean"].replace(0, np.nan)
        ml_df["safety_stock"] = ml_df["safety_stock"].replace(0, np.nan)

        # -------------------------------
        # 🧠 CORE CALCULATIONS
        # -------------------------------
        ml_df["demand_pressure"] = ml_df["rolling_7d_mean"] / (ml_df["safety_stock"] + 1)
        ml_df["doi"] = ml_df["safety_stock"] / (ml_df["rolling_7d_mean"] + 1)

        # -------------------------------
        # ✅ REORDER LOGIC
        # -------------------------------
        ml_df["reorder_flag"] = (
            (ml_df["doi"] < 3) |
            (ml_df["rolling_7d_mean"] > ml_df["reorder_point"])
        )

        # -------------------------------
        # 🚨 RISK CLASSIFICATION (PURE DOI)
        # -------------------------------
        def classify_risk(row):
            if pd.isna(row["doi"]):
                return "Unknown"
            elif row["doi"] < 2:
                return "High Risk"
            elif row["doi"] < 5:
                return "Medium Risk"
            else:
                return "Low Risk"

        ml_df["risk_level"] = ml_df.apply(classify_risk, axis=1)

        # -------------------------------
        # CLEAN DATA FOR KPIs
        # -------------------------------
        ml_df_clean = ml_df.dropna(subset=["doi"])

        # -------------------------------
        # 📊 KPI METRICS (CORRECT ORDER)
        # -------------------------------
        high_risk = len(ml_df_clean[ml_df_clean["risk_level"] == "High Risk"])
        avg_doi = ml_df_clean["doi"].mean()
        reorder_needed = int(ml_df_clean["reorder_flag"].sum())
        total_cost = ml_df_clean["total_inventory_cost_estimate"].sum()

        cost_display = f"{total_cost:,.0f}" if not pd.isna(total_cost) else "N/A"
        doi_display = f"{avg_doi:.1f} days" if not pd.isna(avg_doi) else "N/A"
        cost_per_product = total_cost / len(ml_df_clean) if len(ml_df_clean) > 0 else 0

        # -------------------------------
        # 📊 VISUALIZATION
        # -------------------------------
        fig = px.scatter(
            ml_df_clean,
            x="rolling_7d_mean",
            y="doi",
            color="risk_level",
            size="volatility_index",
            title="Inventory Risk (Demand vs Coverage Days)",
            color_discrete_map={
                "High Risk": "red",
                "Medium Risk": "orange",
                "Low Risk": "green",
                "Unknown": "gray"
            }
        )

        fig.add_hline(y=3, line_dash="dash", line_color="red")
        fig.add_hline(y=7, line_dash="dash", line_color="green")

        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------
        # 📊 KPI DISPLAY
        # -------------------------------
        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("🔴 High Risk", high_risk)
        col2.metric("📦 Avg DOI", doi_display)
        col3.metric("🔁 Reorder Needed", reorder_needed)
        col4.metric("💰 Total Cost", cost_display)
        col5.metric("💸 Cost/Product", f"{cost_per_product:,.0f}")

        # -------------------------------
        # 📌 REPLENISHMENT
        # -------------------------------
        st.subheader("Replenishment Recommendation")

        if reorder_needed > 0:
            st.error(f"⚠️ {reorder_needed} products need immediate reorder")
            st.info("💡 Suggested Action: Use EOQ to replenish stock")
        else:
            st.success("No immediate reorder required")

        # -------------------------------
        # 🧠 ADVANCED INSIGHTS
        # -------------------------------
        st.subheader("📌 Insights")

        if high_risk > 0:
            st.error("⚠️High stockout risk detected")

        if not pd.isna(avg_doi):
            if avg_doi < 3:
                st.error("⚠️Critical: Inventory coverage < 3 days")
            elif avg_doi < 7:
                st.warning("Moderate inventory coverage")
            else:
                st.success("Healthy inventory coverage")

        # ✅ BETTER STOCK ANALYSIS
        breach_ratio = (
            (ml_df_clean["rolling_7d_mean"] > ml_df_clean["safety_stock"]).mean()
        )

        st.metric("Stock Breach %", f"{breach_ratio*100:.1f}%")

        if breach_ratio > 0.3:
            st.error(" Many products exceeding safety stock → High risk")

        if total_cost > 100000:
            st.warning("💰 High inventory cost → Optimize stock levels")

        # -------------------------------
        # 🔥 TOP CRITICAL PRODUCTS (NEW)
        # -------------------------------
        st.subheader("🚨 Top Critical Products")

        top_critical = ml_df_clean.sort_values("doi").head(5)

        st.dataframe(
            top_critical[["product_id", "doi", "eoq", "rolling_7d_mean"]]
        )

        # -------------------------------
        # 📋 FULL DATA
        # -------------------------------
        st.subheader("Detailed Inventory Data")

        st.dataframe(
            ml_df_clean.style.background_gradient(
                subset=["demand_pressure", "doi"],
                cmap="RdYlGn"
            )
        )

with tab3:

    st.subheader("Supplier Intelligence")

    # -------------------------------
    # 📥 LOAD DATA
    # -------------------------------
    with st.spinner("Loading supplier data..."):
        supplier_df = run_query("""
        SELECT supplier_id,
               AVG(on_time_delivery_rate) as on_time_rate,
               AVG(defect_rate) as defect_rate,
               AVG(avg_lead_time_days) as lead_time,
               AVG(cost_variance_pct) as cost_variance
        FROM weekly_supplier_updates
        GROUP BY supplier_id
        """)

    if supplier_df.empty:
        st.warning("No supplier data available")
    else:
        # -------------------------------
        # 🧠 DATA CLEANING
        # -------------------------------
        supplier_df = supplier_df.replace([np.inf, -np.inf], np.nan)

        # Convert to %
        supplier_df["on_time_rate"] = supplier_df["on_time_rate"] * 100
        supplier_df["defect_rate"] = supplier_df["defect_rate"] * 100
        supplier_df["cost_variance"] = supplier_df["cost_variance"] * 100

        # Fill missing safely
        supplier_df = supplier_df.fillna(supplier_df.median(numeric_only=True))

        # -------------------------------
        # 🧠 NORMALIZATION (CRITICAL FIX)
        # -------------------------------
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()

        cols = ["on_time_rate", "defect_rate", "cost_variance", "lead_time"]
        scaled = scaler.fit_transform(supplier_df[cols])

        scaled_df = pd.DataFrame(scaled, columns=cols)

        # -------------------------------
        # 🧠 PERFORMANCE SCORE (IMPROVED)
        # -------------------------------
        supplier_df["performance_score"] = (
            scaled_df["on_time_rate"] * 0.4 +
            (1 - scaled_df["defect_rate"]) * 0.3 +
            (1 - scaled_df["cost_variance"]) * 0.2 +
            (1 - scaled_df["lead_time"]) * 0.1
        ) * 100

        # -------------------------------
        # 🚨 RISK CLASSIFICATION (HYBRID)
        # -------------------------------
        def classify_supplier(score):
            if score < 60:
                return "High Risk"
            elif score < 80:
                return "Medium Risk"
            else:
                return "Low Risk"

        supplier_df["risk_level"] = supplier_df["performance_score"].apply(classify_supplier)

        # -------------------------------
        # 📊 KPI METRICS
        # -------------------------------
        col1, col2, col3, col4 = st.columns(4)

        total_suppliers = supplier_df["supplier_id"].nunique()
        high_risk = len(supplier_df[supplier_df["risk_level"] == "High Risk"])
        avg_score = supplier_df["performance_score"].mean()
        avg_lead = supplier_df["lead_time"].mean()

        col1.metric("🏭 Suppliers", total_suppliers)
        col2.metric("🔴 High Risk", high_risk)
        col3.metric("⭐ Avg Score", f"{avg_score:.1f}")
        col4.metric("⏱ Avg Lead Time", f"{avg_lead:.1f} days")

        # -------------------------------
        # 📊 SCATTER (FIXED CLUSTERING)
        # -------------------------------
        # Add jitter to avoid overlap
        supplier_df["on_time_jitter"] = supplier_df["on_time_rate"] + np.random.uniform(-0.3, 0.3, len(supplier_df))
        supplier_df["defect_jitter"] = supplier_df["defect_rate"] + np.random.uniform(-0.2, 0.2, len(supplier_df))

        fig = px.scatter(
            supplier_df,
            x="on_time_jitter",
            y="defect_jitter",
            color="risk_level",
            size="performance_score",
            hover_name="supplier_id",
            title="Supplier Performance (On-Time vs Defect Rate)",
            opacity=0.8,
            color_discrete_map={
                "High Risk": "red",
                "Medium Risk": "orange",
                "Low Risk": "green"
            }
        )

        fig.add_vline(x=80, line_dash="dash", line_color="green")
        fig.add_hline(y=5, line_dash="dash", line_color="red")

        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------
        # 🏆 TOP SUPPLIERS
        # -------------------------------
        st.subheader("🏆 Top Performing Suppliers")

        supplier_df["rank"] = supplier_df["performance_score"].rank(ascending=False)

        top_suppliers = supplier_df.sort_values("performance_score", ascending=False).head(5)

        st.dataframe(top_suppliers[[
            "supplier_id",
            "performance_score",
            "rank",
            "on_time_rate",
            "defect_rate"
        ]])

        # -------------------------------
        # ⚠️ HIGH RISK SUPPLIERS
        # -------------------------------
        st.subheader("High Risk Suppliers")

        worst_suppliers = supplier_df.sort_values("performance_score").head(5)

        st.dataframe(worst_suppliers[[
            "supplier_id",
            "performance_score",
            "lead_time",
            "cost_variance"
        ]])

        # -------------------------------
        # 📊 COST vs LEAD TIME (IMPROVED)
        # -------------------------------
        st.subheader("Cost vs Lead Time Analysis")

        fig2 = px.scatter(
            supplier_df,
            x="lead_time",
            y="cost_variance",
            color="risk_level",
            size="performance_score",
            hover_name="supplier_id",
            title="Supplier Efficiency (Lead Time vs Cost Variance)",
            color_discrete_map={
                "High Risk": "red",
                "Medium Risk": "orange",
                "Low Risk": "green"
            }
        )

        st.plotly_chart(fig2, use_container_width=True)

        # -------------------------------
        # 🧠 BUSINESS INSIGHTS (SMART)
        # -------------------------------
        st.subheader("Insights")

        if high_risk > 0:
            st.error("⚠️ Underperforming suppliers detected → review contracts")

        if avg_lead > 10:
            st.warning("High lead time → delay risk")

        if supplier_df["defect_rate"].mean() > 7:
            st.error(" High defect rate → quality issue")

        if supplier_df["cost_variance"].mean() > 10:
            st.warning("Cost fluctuations → financial risk")

        if avg_score > 85:
            st.success("Supplier network performing well")

        # -------------------------------
        # 📋 FULL TABLE
        # -------------------------------
        st.subheader("Supplier Data")

        st.dataframe(
            supplier_df.style.background_gradient(
                subset=["performance_score"],
                cmap="RdYlGn"
            )
        )

with tab4:

    st.header("Supply Chain Performance Dashboard")

    st.info(f'Date Range: {params["start_date"]} to {params["end_date"]}')

    # --------------------------------------------------
    # PERFORMANCE QUERY
    # --------------------------------------------------
    perf_sql = f"""
    SELECT
        DATE_TRUNC('day', actual_delivery_date) AS date,
        COUNT(*) AS total_orders,
        {yes_no_rate("SUM(CASE WHEN on_time = '1' THEN 1 ELSE 0 END)")} AS on_time_rate,
        {yes_no_rate("SUM(CASE WHEN in_full = '1' THEN 1 ELSE 0 END)")} AS in_full_rate,
        {yes_no_rate("SUM(CASE WHEN on_time_in_full = '1' THEN 1 ELSE 0 END)")} AS otif_rate
    FROM fact_order_line
    WHERE actual_delivery_date BETWEEN :start_date AND :end_date
        AND actual_delivery_date IS NOT NULL
    GROUP BY 1
    ORDER BY date;
    """

    df_perf = safe_query(perf_sql, params)

    if df_perf.empty:
        st.error("❌ No performance data available")
    else:
        df_perf["date"] = pd.to_datetime(df_perf["date"])

        for col in ["total_orders", "on_time_rate", "in_full_rate", "otif_rate"]:
            df_perf[col] = pd.to_numeric(df_perf[col], errors="coerce").fillna(0)

        # Moving Average
        df_perf["otif_ma"] = df_perf["otif_rate"].rolling(7, min_periods=1).mean()

        # KPIs
        avg_otif = df_perf["otif_rate"].mean()
        avg_on_time = df_perf["on_time_rate"].mean()
        avg_in_full = df_perf["in_full_rate"].mean()
        total_orders = df_perf["total_orders"].sum()

        latest = df_perf.iloc[-1]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("OTIF (Avg)", f"{avg_otif:.1f}%", f"{latest['otif_rate'] - avg_otif:+.1f}% vs latest")
        col2.metric("On-Time", f"{avg_on_time:.1f}%", f"{latest['on_time_rate'] - avg_on_time:+.1f}%")
        col3.metric("In-Full", f"{avg_in_full:.1f}%", f"{latest['in_full_rate'] - avg_in_full:+.1f}%")
        col4.metric("Total Orders", f"{int(total_orders):,}")

        st.markdown("---")

        # --------------------------------------------------
        # 📈 TRENDS (ADDED 🔥)
        # --------------------------------------------------
        st.subheader("Performance Trends")

        fig_trend = go.Figure()

        fig_trend.add_trace(go.Scatter(x=df_perf["date"], y=df_perf["otif_rate"], name="OTIF"))
        fig_trend.add_trace(go.Scatter(x=df_perf["date"], y=df_perf["otif_ma"], name="OTIF Trend", line=dict(dash="dot")))

        fig_trend.update_layout(template="plotly_white", height=400)

        st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("---")

        # --------------------------------------------------
        # 🎯 CUSTOMER RADAR (FIXED)
        # --------------------------------------------------
        st.subheader("Top Customers Comparison")

        customer_sql = f"""
        SELECT
            c.customer_name,
            {yes_no_rate("SUM(CASE WHEN f.on_time = '1' THEN 1 ELSE 0 END)")} AS on_time,
            {yes_no_rate("SUM(CASE WHEN f.in_full = '1' THEN 1 ELSE 0 END)")} AS in_full,
            {yes_no_rate("SUM(CASE WHEN f.on_time_in_full = '1' THEN 1 ELSE 0 END)")} AS otif,
            COUNT(*) as orders
        FROM fact_order_line f
        LEFT JOIN dim_customers c ON f.customer_id = c.customer_id
        WHERE f.actual_delivery_date BETWEEN :start_date AND :end_date
        GROUP BY c.customer_name
        HAVING COUNT(*) > 10
        ORDER BY otif DESC
        LIMIT 5
        """

        df_cust = safe_query(customer_sql, params)

        if not df_cust.empty:
            max_orders = df_cust["orders"].max()

            fig = go.Figure()

            for _, r in df_cust.iterrows():
                volume_score = (r["orders"] / max_orders) * 100

                fig.add_trace(go.Scatterpolar(
                    r=[r["on_time"], r["in_full"], r["otif"], volume_score],
                    theta=["On-Time", "In-Full", "OTIF", "Volume"],
                    fill='toself',
                    name=r["customer_name"]
                ))

            fig.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=500)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # --------------------------------------------------
        # 🔥 HEATMAP (FIXED)
        # --------------------------------------------------
        st.subheader("OTIF Heatmap")

        heatmap_sql = f"""
        SELECT
            EXTRACT(DOW FROM actual_delivery_date) AS dow,
            EXTRACT(WEEK FROM actual_delivery_date) AS week,
            {yes_no_rate("SUM(CASE WHEN on_time_in_full = '1' THEN 1 ELSE 0 END)")} AS otif
        FROM fact_order_line
        WHERE actual_delivery_date BETWEEN :start_date AND :end_date
        GROUP BY 1,2
        """

        df_heat = safe_query(heatmap_sql, params)

        if not df_heat.empty:
            pivot = df_heat.pivot(index="dow", columns="week", values="otif").clip(0, 100)

            day_map = {0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat"}
            pivot.index = pivot.index.map(day_map)

            fig = go.Figure(go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="RdYlGn",
                zmin=0,
                zmax=100,
                zmid=75
            ))

            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # --------------------------------------------------
        # 📦 BOX PLOT (CLEANED)
        # --------------------------------------------------
        st.subheader("Delivery Time Distribution")

        box_sql = """
        SELECT
            p.category,
            (f.actual_delivery_date - f.order_placement_date) AS days,
            CASE WHEN f.on_time = '1' THEN 'On Time' ELSE 'Late' END AS timing
        FROM fact_order_line f
        LEFT JOIN dim_products p ON f.product_id = p.product_id
        WHERE f.actual_delivery_date BETWEEN :start_date AND :end_date
        AND (f.actual_delivery_date - f.order_placement_date) BETWEEN 1 AND 45
        """

        df_box = safe_query(box_sql, params)

        if not df_box.empty:
            fig = px.box(df_box, x="category", y="days", color="timing")
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # --------------------------------------------------
        # 🔁 SANKEY (BUSINESS FIXED)
        # --------------------------------------------------
        st.subheader("Order Flow")

        sankey_sql = """
        SELECT
            p.category,
            CASE 
                WHEN f.on_time_in_full = '1' THEN 'Delivered Successfully'
                WHEN f.on_time = '0' AND f.in_full = '1' THEN 'Late Delivery'
                WHEN f.in_full = '0' AND f.on_time = '1' THEN 'Partial Delivery'
                ELSE 'Critical Issue'
            END AS status,
            COUNT(*) as cnt
        FROM fact_order_line f
        LEFT JOIN dim_products p ON f.product_id = p.product_id
        WHERE f.actual_delivery_date BETWEEN :start_date AND :end_date
        GROUP BY 1,2
        """

        df_sankey = safe_query(sankey_sql, params)

        if not df_sankey.empty:
            labels = list(df_sankey["category"].unique()) + list(df_sankey["status"].unique())

            source, target, value = [], [], []

            for _, r in df_sankey.iterrows():
                source.append(labels.index(r["category"]))
                target.append(labels.index(r["status"]))
                value.append(r["cnt"])

            fig = go.Figure(go.Sankey(
                node=dict(label=labels),
                link=dict(source=source, target=target, value=value)
            ))

            fig.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)

with tab5:

    st.header("Logistics & Warehouse Intelligence")

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------
    df_wh = safe_query("""
    SELECT 
        m.product_id,
        m.warehouse,
        w.warehouse_name,
        w.city,
        w.latitude,
        w.longitude,
        m.net_stock_position,
        m.stock_status,
        m.suggested_transfer_qty,
        m.estimated_transfer_value
    FROM multi_warehouse_optimization m
    LEFT JOIN dim_warehouses w 
    ON m.warehouse = w.warehouse_id
    """)

    if df_wh is None or df_wh.empty:
        st.warning("No warehouse data available")
    else:

        # --------------------------------------------------
        # DATA CLEANING
        # --------------------------------------------------
        df_wh = df_wh.copy()

        df_wh["net_stock_position"] = pd.to_numeric(df_wh["net_stock_position"], errors="coerce").fillna(0)
        df_wh["suggested_transfer_qty"] = pd.to_numeric(df_wh["suggested_transfer_qty"], errors="coerce").fillna(0)
        df_wh["estimated_transfer_value"] = pd.to_numeric(df_wh["estimated_transfer_value"], errors="coerce").fillna(0)

        # --------------------------------------------------
        # STOCK TYPE (SOURCE OF TRUTH)
        # --------------------------------------------------
        df_wh["stock_type"] = df_wh["net_stock_position"].apply(
            lambda x: "Surplus" if x > 0 else ("Shortage" if x < 0 else "Balanced")
        )

        # --------------------------------------------------
        # KPI (FIXED)
        # --------------------------------------------------
        total_movement = df_wh["suggested_transfer_qty"].sum()
        transfer_value = df_wh["estimated_transfer_value"].sum()

        shortage = df_wh[df_wh["net_stock_position"] < 0]["warehouse"].nunique()
        surplus = df_wh[df_wh["net_stock_position"] > 0]["warehouse"].nunique()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Movement", f"{total_movement:,.0f}")
        col2.metric("Shortage Warehouses", shortage)
        col3.metric("Surplus Warehouses", surplus)
        col4.metric("Transfer Value", f"{transfer_value:,.0f}")

        st.markdown("---")

        # --------------------------------------------------
        # STOCK DISTRIBUTION (FIXED)
        # --------------------------------------------------
        st.subheader("Stock Distribution")

        df_group = (
            df_wh.groupby(["warehouse_name", "stock_type"])["net_stock_position"]
            .sum()
            .reset_index()
        )

        df_group["value"] = df_group["net_stock_position"].abs()

        fig = px.bar(
            df_group,
            x="warehouse_name",
            y="value",
            color="stock_type",
            barmode="group"
        )

        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # --------------------------------------------------
        # PIE (FIXED - MAGNITUDE BASED)
        # --------------------------------------------------
        st.subheader("Stock Status Breakdown")

        df_temp = df_wh.copy()
        df_temp["magnitude"] = df_temp["net_stock_position"].abs()

        status_df = (
            df_temp.groupby("stock_type")["magnitude"]
            .sum()
            .reset_index()
        )

        fig = px.pie(status_df, names="stock_type", values="magnitude")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # --------------------------------------------------
        # 🔁 TRANSFER FLOW
        # --------------------------------------------------
        st.subheader("Recommended Transfers")

        df_transfer = safe_query("""
        SELECT 
            from_warehouse,
            to_warehouse,
            SUM(allocated_qty) as allocated_qty,
            SUM(total_transfer_cost) as total_cost
        FROM network_transfer_plan
        GROUP BY 1,2
        """)

        if df_transfer is not None and not df_transfer.empty:

            warehouse_map = (
                df_wh[["warehouse", "warehouse_name"]]
                .drop_duplicates()
                .set_index("warehouse")["warehouse_name"]
                .to_dict()
            )

            df_transfer["from_name"] = df_transfer["from_warehouse"].map(warehouse_map)
            df_transfer["to_name"] = df_transfer["to_warehouse"].map(warehouse_map)

            df_transfer["route"] = (
                df_transfer["from_name"].fillna("Unknown") +
                " → " +
                df_transfer["to_name"].fillna("Unknown")
            )

            fig = px.bar(
                df_transfer,
                x="route",
                y="allocated_qty",
                color="route",
                title="Transfer Flow Between Warehouses"
            )

            fig.update_layout(template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(df_transfer)

        else:
            st.info("No transfer data available")

        st.markdown("---")

        # --------------------------------------------------
        # 🌍 MAP (FIXED SCALING)
        # --------------------------------------------------
        st.subheader("Warehouse Network with Transfer Flow")

        df_all_wh = safe_query("""
        SELECT warehouse_id as warehouse, warehouse_name, latitude, longitude
        FROM dim_warehouses
        """)

        df_map = df_all_wh.copy()

        if not df_map.empty:

            df_map["latitude"] = pd.to_numeric(df_map["latitude"], errors="coerce")
            df_map["longitude"] = pd.to_numeric(df_map["longitude"], errors="coerce")

            df_stock = (
                df_wh.groupby("warehouse")["net_stock_position"]
                .sum()
                .reset_index()
            )

            df_map = df_map.merge(df_stock, on="warehouse", how="left")
            df_map["net_stock_position"] = df_map["net_stock_position"].fillna(0)

            df_map["stock_type"] = df_map["net_stock_position"].apply(
                lambda x: "Surplus" if x > 0 else ("Shortage" if x < 0 else "Balanced")
            )

            df_map["size"] = df_map["net_stock_position"].abs() + 50

            fig = px.scatter_mapbox(
                df_map,
                lat="latitude",
                lon="longitude",
                color="stock_type",
                size="size",
                hover_name="warehouse_name",
                zoom=4,
                height=600
            )

            if df_transfer is not None and not df_transfer.empty:

                coords = df_map.set_index("warehouse")[["latitude", "longitude"]].to_dict("index")
                max_qty = df_transfer["allocated_qty"].max()

                for _, row in df_transfer.iterrows():

                    src = row["from_warehouse"]
                    dst = row["to_warehouse"]

                    if src in coords and dst in coords:

                        lat1, lon1 = coords[src]["latitude"], coords[src]["longitude"]
                        lat2, lon2 = coords[dst]["latitude"], coords[dst]["longitude"]

                        width = max(2, (row["allocated_qty"] / max_qty) * 10)

                        fig.add_trace(
                            go.Scattermapbox(
                                lat=[lat1, lat2],
                                lon=[lon1, lon2],
                                mode="lines",
                                line=dict(width=width, color="green"),
                                opacity=0.6
                            )
                        )

                        fig.add_trace(
                            go.Scattermapbox(
                                lat=[(lat1+lat2)/2],
                                lon=[(lon1+lon2)/2],
                                mode="text",
                                text=["→"],
                                showlegend=False
                            )
                        )

                        fig.add_trace(
                            go.Scattermapbox(
                                lat=[lat2],
                                lon=[lon2],
                                mode="text",
                                text=["🚚"],
                                showlegend=False
                            )
                        )

            fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("No valid location data available")

        st.markdown("---")

        # --------------------------------------------------
        # INSIGHTS (FIXED)
        # --------------------------------------------------
        st.subheader("Logistics Insights")

        if shortage > surplus:
            st.error("⚠️ Network understocked → Immediate redistribution needed")

        elif surplus > shortage:
            st.warning("📦 Excess inventory present → Optimize allocation")

        else:
            st.success("Balanced network")

        if transfer_value > 0:
            st.info("Monitor transfer costs to optimize logistics")

        st.markdown("---")

        # --------------------------------------------------
        # TABLE
        # --------------------------------------------------
        st.subheader("Warehouse Details")

        def highlight(val):
            if val < 0:
                return "background-color:#ff4d4f; color:white;"
            elif val > 0:
                return "background-color:#52c41a; color:white;"
            return "background-color:#262730; color:white;"

        st.dataframe(
            df_wh.style.map(highlight, subset=["net_stock_position"])
        )
with tab6:

    alerts_df = run_query("""
    SELECT alert_time,alert_type,entity_name,detail
    FROM alert_log
    ORDER BY alert_time DESC
    LIMIT 10
    """)

    for _,row in alerts_df.iterrows():

        if row["alert_type"]=="CRITICAL":
            st.error(f"{row['entity_name']} : {row['detail']}")

        elif row["alert_type"]=="WARNING":
            st.warning(f"{row['entity_name']} : {row['detail']}")

        else:
            st.info(f"{row['entity_name']} : {row['detail']}")