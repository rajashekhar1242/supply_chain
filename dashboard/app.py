import os
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# -----------------------------------------------------------------------------
#  PATH SETUP
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from analytics.pipeline import run_pipeline
except ImportError:
    def run_pipeline():
        pass


# -----------------------------------------------------------------------------
#  STREAMLIT CONFIG + STYLES
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Supply Chain Analytics Dashboard", layout="wide")
st.markdown(
    """
    <style>
        .stApp { background-color: #121212; color: #FFFFFF; }
        h1, h2, h3, h4 { color: #FFD700; }
        div[data-testid="stMetricValue"] { color: #00FFB3 !important; }
        .stButton > button {
            background-color: #FF4B4B; color: white;
            border-radius: 8px; padding: 8px 20px; font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st_autorefresh(interval=60 * 1000, key="refresh_dashboard")
if "refresh_counter" not in st.session_state:
    st.session_state.refresh_counter = 0


# -----------------------------------------------------------------------------
#  DATABASE CONNECTION
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_engine():
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASS")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "postgres")

    if not all([db_user, db_pass, db_host]):
        raise RuntimeError("Database credentials missing from environment variables.")

    engine = create_engine(
        f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}",
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=3600,
    )

    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine


def run_pipeline_and_refresh():
    run_pipeline()
    st.session_state.refresh_counter += 1


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


def add_trend_columns(df: pd.DataFrame, col: str, window: int = 7) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    df = df.sort_values("date").copy()
    df[f"{col}_ma"] = df[col].rolling(window=window, min_periods=1).mean()
    df[f"{col}_std"] = df[col].rolling(window=window, min_periods=1).std()
    df[f"{col}_zscore"] = (df[col] - df[f"{col}_ma"]) / (df[f"{col}_std"] + 1e-6)
    return df


def safe_float(value, default=0.0):
    """Safely convert value to float, handling None/NaN."""
    try:
        if pd.isna(value) or value is None:
            return default
        return float(value)
    except:
        return default


# -----------------------------------------------------------------------------
#  SIDEBAR FILTERS
# -----------------------------------------------------------------------------
engine = get_engine()
with st.sidebar:
    st.header("⚙️ Filters")

    default_start = date.today() - timedelta(days=30)
    default_end = date.today()

    start_date = st.date_input("Start Date", value=default_start)
    end_date = st.date_input("End Date", value=default_end)
    if start_date > end_date:
        st.warning("Start date cannot be after end date. Resetting.")
        start_date, end_date = default_start, default_end

    st.markdown("---")
    st.button("Run Analytics Now", on_click=run_pipeline_and_refresh)
    st.info("Auto-refresh every 60s ⏱️")
    
    st.markdown("---")
    st.markdown("**Active Filters:**")
    st.text(f"📅 {start_date} to {end_date}")


params = {"start_date": start_date, "end_date": end_date}


# -----------------------------------------------------------------------------
#  TABS
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "Customer Risk",
    "Forecasting",
    "KPIs",
    "Latest Analytics",
    "Performance Dashboard",
    "Alerts",
    "Trends",
    "Configuration",
    "Executive Summary",
])


# -----------------------------------------------------------------------------
#  TAB 1 — CUSTOMER RISK (Dark-Themed, Fixed Table)
# -----------------------------------------------------------------------------
with tabs[0]:
    st.header("1️⃣ Customer Risk Analysis")

    risk_sql = """
        SELECT c.customer_name,
               k.customer_id,
               k.prediction_date AS date,
               k.risk_score,
               k.risk_label,
               k.created_at AS updated_at
        FROM kpi_risk_predictions k
        JOIN dim_customers c ON k.customer_id = c.customer_id
        WHERE k.prediction_date BETWEEN :start_date AND :end_date
        ORDER BY k.risk_score DESC, k.created_at DESC
        LIMIT 500;
    """
    df_risk = safe_query(risk_sql, params)

    if df_risk.empty:
        st.warning("No customer risk data available for selected period.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        total_customers = df_risk['customer_id'].nunique()
        at_risk = len(df_risk[df_risk['risk_label'] == 'At Risk'])
        safe_count = len(df_risk[df_risk['risk_label'] == 'Safe'])
        avg_risk = df_risk['risk_score'].mean()

        col1.metric("📊 Total Customers", f"{total_customers:,}")
        col2.metric("⚠️ At Risk", f"{at_risk:,}")
        col3.metric("✅ Safe", f"{safe_count:,}")
        col4.metric("📈 Avg Risk Score", f"{avg_risk:.2f}")

        st.markdown("---")
        st.dataframe(df_risk)

        # --- Top 20 customers chart ---
        fig_scores = px.bar(
            df_risk.nlargest(20, 'risk_score'),
            x="customer_name",
            y="risk_score",
            color="risk_label",
            color_discrete_map={"At Risk": "#FF4B4B", "Safe": "#00FF7F"},
            template="plotly_dark",
            title="Top 20 Customers by Risk Score",
        )
        fig_scores.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_scores, use_container_width=True)

        # --- Risk category dashboard ---
        st.subheader("📊 Risk Category Distribution Dashboard")
        risk_group = (
            df_risk.groupby("risk_label")["customer_id"]
            .count()
            .reset_index()
            .rename(columns={"risk_label": "risk_category", "customer_id": "customer_count"})
        )

        fig_risk = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Risk Category Distribution",
                "Customer Count by Risk Category",
                "Risk Category Breakdown",
                "High-Risk Customers",
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "pie"}, {"type": "table"}]],
        )

        # --- Bar: Risk Distribution ---
        fig_risk.add_trace(
            go.Bar(
                x=risk_group["risk_category"],
                y=risk_group["customer_count"],
                name="Risk Distribution",
                marker_color="#1f77b4",
                text=risk_group["customer_count"],
                textposition="auto",
            ),
            row=1, col=1,
        )

        # --- Bar: Customer Count ---
        fig_risk.add_trace(
            go.Bar(
                x=risk_group["customer_count"],
                y=risk_group["risk_category"],
                name="Customer Count",
                marker_color="#ff7f0e",
                orientation="h",
                text=risk_group["customer_count"],
                textposition="auto",
            ),
            row=1, col=2,
        )

        # --- Pie Chart ---
        fig_risk.add_trace(
            go.Pie(
                labels=risk_group["risk_category"],
                values=risk_group["customer_count"],
                name="Risk Categories",
                marker=dict(colors=["#FF4B4B", "#00FF7F"]),
            ),
            row=2, col=1,
        )

        # --- Table: High-Risk Customers (FIXED COLOR SCHEME) ---
        high_risk = df_risk[df_risk["risk_label"] == "At Risk"].nlargest(15, 'risk_score')
        if not high_risk.empty:
            fig_risk.add_trace(
                go.Table(
                    header=dict(
                        values=["Customer", "Risk Score", "Updated At"],
                        fill_color="#003366",
                        font=dict(color="white", size=12),
                        align="left",
                    ),
                    cells=dict(
                        values=[
                            high_risk["customer_name"],
                            np.round(high_risk["risk_score"], 4),
                            high_risk["updated_at"].astype(str),
                        ],
                        fill_color="#1E1E1E",
                        font=dict(color="white", size=11),
                        align="left",
                    ),
                ),
                row=2, col=2,
            )

        # --- Final Layout ---
        fig_risk.update_layout(
            title_text="📊 Customer Risk Distribution Overview",
            height=820,
            showlegend=True,
            template="plotly_dark",
            title_font=dict(size=20, color="white"),
            font=dict(color="white"),
        )

        st.plotly_chart(fig_risk, use_container_width=True)



# ----------------------------------------------------------------------------- 
#  TAB 2 — FORECASTING (WITH SUNBURST CHART) 
# ----------------------------------------------------------------------------- 
with tabs[1]:
    st.header("2️⃣ Demand Forecasting Dashboard")

    forecast_sql = """
        SELECT 
          p.product_id,
          d.product_name,
          d.category,
          p.forecast_demand,
          p.forecast_accuracy,
          i.stock_level
        FROM policy_recommendations p
        LEFT JOIN dim_products d 
        ON p.product_id = d.product_id
        LEFT JOIN derived_inventory i 
        ON p.product_id = i.product_id
        WHERE p.forecast_demand IS NOT NULL
        ORDER BY p.forecast_demand DESC;
    """
    df_forecast = safe_query(forecast_sql)

    if df_forecast.empty:
        st.warning("No forecast data available.")
    else:
        # Fill missing numeric values with 0
        df_forecast['forecast_demand'] = df_forecast['forecast_demand'].fillna(0)
        df_forecast['stock_level'] = df_forecast['stock_level'].fillna(0)
        df_forecast['forecast_accuracy'] = df_forecast['forecast_accuracy'].fillna(0)
        
        # Fill missing product info
        df_forecast['product_name'] = df_forecast['product_name'].fillna('Unknown Product')
        df_forecast['category'] = df_forecast['category'].fillna('Uncategorized')

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        total_forecast = df_forecast['forecast_demand'].sum()
        total_stock = df_forecast['stock_level'].sum()
        avg_accuracy = df_forecast['forecast_accuracy'].mean()
        stockout_risk = len(df_forecast[df_forecast['stock_level'] < df_forecast['forecast_demand']])

        col1.metric("📦 Total Forecast", f"{total_forecast:,.0f}")
        col2.metric("📊 Current Stock", f"{total_stock:,.0f}")
        col3.metric("🎯 Avg Accuracy", f"{avg_accuracy:.1f}%")
        col4.metric("⚠️ Stockout Risk", f"{stockout_risk}")

        st.markdown("---")

        # Sunburst Chart: Category → Product → Stock Status
        st.subheader("☀️ Product Hierarchy: Category → Product → Stock Status")
        sunburst_df = df_forecast.copy()
        sunburst_df['status'] = sunburst_df.apply(
            lambda x: 'Sufficient Stock' if x['stock_level'] >= x['forecast_demand'] else 'Low Stock',
            axis=1
        )

        fig_sunburst = px.sunburst(
            sunburst_df.head(30),
            path=['category', 'product_name', 'status'],
            values='forecast_demand',
            color='forecast_accuracy',
            color_continuous_scale='RdYlGn',
            title="☀️ Forecast Distribution: Category → Product → Stock Status"
        )
        fig_sunburst.update_layout(template="plotly_dark", height=600)
        st.plotly_chart(fig_sunburst, use_container_width=True)

        st.markdown("---")

        # Bar chart: Total forecasted demand by category
        category_summary = (
            df_forecast.groupby("category")["forecast_demand"]
            .sum()
            .reset_index()
            .sort_values("forecast_demand", ascending=False)
        )
        fig_cat = px.bar(
            category_summary,
            x="category",
            y="forecast_demand",
            color="forecast_demand",
            color_continuous_scale="Blues",
            title="📦 Total Forecasted Demand by Category",
            labels={"forecast_demand": "Forecasted Units"},
        )
        fig_cat.update_layout(template="plotly_dark")
        st.plotly_chart(fig_cat, use_container_width=True)

        # Stockout risk table
        st.subheader("⚠️ Products at Stockout Risk")
        stockout_df = df_forecast[df_forecast['stock_level'] < df_forecast['forecast_demand']].copy()
        if not stockout_df.empty:
            stockout_df['shortage'] = stockout_df['forecast_demand'] - stockout_df['stock_level']
            st.error(f"⚠️ {len(stockout_df)} products at risk of stockout!")
            st.dataframe(
                stockout_df[['product_name', 'category', 'forecast_demand', 'stock_level', 'shortage', 'recommended_order']],
                use_container_width=True
            )
        else:
            st.success("✅ No immediate stockout risks!")


with tabs[2]:
    st.header("3️⃣ KPI Analysis")

    kpi_sql = """
        SELECT 
            c.customer_name,
            f.customer_id,
            COUNT(*) AS total_orders,
            ROUND(COALESCE(SUM(CASE WHEN on_time = 1 THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 0), 2) AS on_time_rate,
            ROUND(COALESCE(SUM(CASE WHEN in_full = '1' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 0), 2) AS in_full_rate,
            ROUND(COALESCE(SUM(CASE WHEN otif = '1' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 0), 2) AS otif_rate,
            MAX(order_placement_date) AS last_order_date
        FROM fact_aggregate f
        JOIN dim_customers c ON f.customer_id = c.customer_id
        WHERE f.order_placement_date BETWEEN :start_date AND :end_date
        GROUP BY c.customer_name, f.customer_id
        HAVING COUNT(*) >= 5
        ORDER BY otif_rate DESC
        LIMIT 100;
    """
    df_kpi = safe_query(kpi_sql, params)

    if df_kpi.empty:
        st.warning("No KPI data available for selected period.")
    else:
        # Convert to numeric
        df_kpi['on_time_rate'] = pd.to_numeric(df_kpi['on_time_rate'], errors='coerce')
        df_kpi['in_full_rate'] = pd.to_numeric(df_kpi['in_full_rate'], errors='coerce')
        df_kpi['otif_rate'] = pd.to_numeric(df_kpi['otif_rate'], errors='coerce')
        df_kpi['total_orders'] = pd.to_numeric(df_kpi['total_orders'], errors='coerce')
        
        col1, col2, col3, col4 = st.columns(4)
        total_orders = int(df_kpi['total_orders'].sum())
        avg_on_time = safe_float(df_kpi['on_time_rate'].mean())
        avg_in_full = safe_float(df_kpi['in_full_rate'].mean())
        avg_otif = safe_float(df_kpi['otif_rate'].mean())
        
        col1.metric("📦 Total Orders", f"{total_orders:,}")
        col2.metric("⏰ Avg On-Time", f"{avg_on_time:.1f}%")
        col3.metric("📦 Avg In-Full", f"{avg_in_full:.1f}%")
        col4.metric("🎯 Avg OTIF", f"{avg_otif:.1f}%")
        
        st.markdown("---")
        
        # ⭐ Radar Chart for Top 5 Customers
        st.subheader("🎯 Multi-Metric Customer Comparison (Top 5)")
        
        top_5_customers = df_kpi.nlargest(5, 'otif_rate')
        
        if len(top_5_customers) >= 2:
            fig_radar = go.Figure()
            
            categories = ['On-Time %', 'In-Full %', 'OTIF %', 'Order Volume']
            
            for idx, row in top_5_customers.iterrows():
                # Normalize order volume to 0-100 scale
                max_orders = df_kpi['total_orders'].max()
                order_score = (row['total_orders'] / max_orders * 100) if max_orders > 0 else 0
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row['on_time_rate'], row['in_full_rate'], row['otif_rate'], order_score],
                    theta=categories,
                    fill='toself',
                    name=row['customer_name'][:20]
                ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                title="🎯 Top 5 Customers: Multi-Metric Performance",
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        st.markdown("---")
        
        st.dataframe(
            df_kpi.style.background_gradient(subset=['on_time_rate', 'in_full_rate', 'otif_rate'], cmap='RdYlGn'),
            use_container_width=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top 10 Customers by OTIF")
            top_10 = df_kpi.nlargest(10, 'otif_rate')
            fig_top = px.bar(
                top_10,
                x='customer_name',
                y='otif_rate',
                title="Top Performers",
                template="plotly_dark",
                color='otif_rate',
                color_continuous_scale='Greens'
            )
            fig_top.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_top, use_container_width=True)
        
        with col2:
            st.subheader("⚠️ Bottom 10 Customers by OTIF")
            bottom_10 = df_kpi.nsmallest(10, 'otif_rate')
            fig_bottom = px.bar(
                bottom_10,
                x='customer_name',
                y='otif_rate',
                title="Needs Improvement",
                template="plotly_dark",
                color='otif_rate',
                color_continuous_scale='Reds'
            )
            fig_bottom.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bottom, use_container_width=True)


# -----------------------------------------------------------------------------
#  TAB 4 — LATEST ANALYTICS
# -----------------------------------------------------------------------------
with tabs[3]:
    st.header("4️⃣ Latest Analytics Results")
    
    alerts_sql = """
        SELECT id,
               alert_time AS updated_at,
               alert_type,
               entity_name,
               detail
        FROM alert_log
        WHERE alert_time BETWEEN :start_date AND :end_date
        ORDER BY alert_time DESC
        LIMIT 100;
    """
    df_alert_log = safe_query(alerts_sql, params)

    if df_alert_log.empty:
        st.info("No analytics alerts found for selected period.")
    else:
        col1, col2, col3 = st.columns(3)
        total_alerts = len(df_alert_log)
        alert_types = df_alert_log['alert_type'].nunique()
        latest = df_alert_log['updated_at'].max()
        
        col1.metric("🚨 Total Alerts", f"{total_alerts:,}")
        col2.metric("📊 Alert Types", f"{alert_types}")
        col3.metric("🕐 Latest", latest.strftime('%Y-%m-%d %H:%M') if pd.notnull(latest) else "N/A")
        
        st.markdown("---")
        st.dataframe(df_alert_log)
        st.download_button(
            "⬇️ Download Alerts",
            df_alert_log.to_csv(index=False),
            file_name=f"alerts_{start_date}_{end_date}.csv",
            mime="text/csv",
        )


# =============================================================================
#  TAB 5 — PERFORMANCE DASHBOARD (FULLY CORRECTED VERSION)
# =============================================================================
with tabs[4]:
    st.header("5️⃣ Supply Chain Performance Dashboard")
    
    # Debug: Show selected date range
    st.info(f'📅 Date Range: {params["start_date"]} to {params["end_date"]}')
    
    # Main performance query
    perf_sql = f"""
    SELECT
        DATE_TRUNC('day', actual_delivery_date) AS date,
        COUNT(*) AS total_orders,
        {yes_no_rate('SUM(CASE WHEN "On Time" = \'1\' THEN 1 ELSE 0 END)')} AS on_time_rate,
        {yes_no_rate('SUM(CASE WHEN "In Full" = \'1\' THEN 1 ELSE 0 END)')} AS in_full_rate,
        {yes_no_rate('SUM(CASE WHEN "On Time In Full" = \'1\' THEN 1 ELSE 0 END)')} AS otif_rate
    FROM fact_order_line
    WHERE actual_delivery_date BETWEEN :start_date AND :end_date
        AND actual_delivery_date IS NOT NULL
    GROUP BY 1
    HAVING COUNT(*) > 0
    ORDER BY date;
    """
    
    df_perf = safe_query(perf_sql, params)
    
    if df_perf.empty:
        st.error('❌ No daily performance data available for the selected range.')
        st.warning('💡 Try expanding your date range or check if data exists in the database.')
        
        # Show available date range
        date_check_sql = """
        SELECT 
            MIN(actual_delivery_date) as min_date,
            MAX(actual_delivery_date) as max_date,
            COUNT(*) as total_records
        FROM fact_order_line
        WHERE actual_delivery_date IS NOT NULL;
        """
        df_date_check = safe_query(date_check_sql, {})
        if not df_date_check.empty:
            st.info(f'📊 Available data range: {df_date_check.iloc[0]["min_date"]} to {df_date_check.iloc[0]["max_date"]} ({df_date_check.iloc[0]["total_records"]:,} records)')
    else:
        # Convert data types properly
        df_perf['date'] = pd.to_datetime(df_perf['date'])
        df_perf['total_orders'] = pd.to_numeric(df_perf['total_orders'], errors='coerce').fillna(0).astype(int)
        df_perf['on_time_rate'] = pd.to_numeric(df_perf['on_time_rate'], errors='coerce').fillna(0)
        df_perf['in_full_rate'] = pd.to_numeric(df_perf['in_full_rate'], errors='coerce').fillna(0)
        df_perf['otif_rate'] = pd.to_numeric(df_perf['otif_rate'], errors='coerce').fillna(0)
        
        # Add trend columns
        df_perf = add_trend_columns(df_perf, 'otif_rate')
        df_perf = add_trend_columns(df_perf, 'on_time_rate')
        df_perf = add_trend_columns(df_perf, 'in_full_rate')
        
        # ✅ FIXED: Calculate overall averages instead of using last day
        avg_otif = df_perf['otif_rate'].mean()
        avg_on_time = df_perf['on_time_rate'].mean()
        avg_in_full = df_perf['in_full_rate'].mean()
        total_orders_sum = df_perf['total_orders'].sum()
        
        # Get latest day's performance
        latest = df_perf.iloc[-1]
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric(
            '🎯 OTIF Rate (Avg)', 
            f'{avg_otif:.1f}%',
            delta=f'{(latest["otif_rate"] - avg_otif):+.1f}% last day',
            help=f'Overall average: {avg_otif:.1f}% | Latest: {latest["otif_rate"]:.1f}%'
        )
        col2.metric(
            '⏰ On-Time Rate (Avg)', 
            f'{avg_on_time:.1f}%',
            delta=f'{(latest["on_time_rate"] - avg_on_time):+.1f}% last day',
            help=f'Overall average: {avg_on_time:.1f}% | Latest: {latest["on_time_rate"]:.1f}%'
        )
        col3.metric(
            '📦 In-Full Rate (Avg)', 
            f'{avg_in_full:.1f}%',
            delta=f'{(latest["in_full_rate"] - avg_in_full):+.1f}% last day',
            help=f'Overall average: {avg_in_full:.1f}% | Latest: {latest["in_full_rate"]:.1f}%'
        )
        col4.metric('📈 Total Orders', f'{total_orders_sum:,}')
        
        st.markdown('---')
        
        # ⭐ IMPROVED: Performance Heatmap
        st.subheader('🔥 Performance Heatmap: OTIF by Day of Week')
        
        heatmap_sql = f"""
            SELECT
                EXTRACT(DOW FROM actual_delivery_date)::int AS day_of_week,
                EXTRACT(WEEK FROM actual_delivery_date)::int AS week_number,
                {yes_no_rate('SUM(CASE WHEN "On Time In Full" = \'1\' THEN 1 ELSE 0 END)')} AS otif_rate,
                COUNT(*) as order_count
            FROM fact_order_line
            WHERE actual_delivery_date BETWEEN :start_date AND :end_date
                AND actual_delivery_date IS NOT NULL
            GROUP BY 1, 2
            HAVING COUNT(*) >= 5
            ORDER BY 2, 1;
        """
        df_heatmap = safe_query(heatmap_sql, params)
        
        if not df_heatmap.empty and len(df_heatmap) >= 5:
            try:
                # Convert to proper types
                df_heatmap['day_of_week'] = pd.to_numeric(df_heatmap['day_of_week'], errors='coerce')
                df_heatmap['week_number'] = pd.to_numeric(df_heatmap['week_number'], errors='coerce')
                df_heatmap['otif_rate'] = pd.to_numeric(df_heatmap['otif_rate'], errors='coerce')
                
                # Remove any rows with null values
                df_heatmap = df_heatmap.dropna()
                
                # Filter valid day of week (0-6)
                df_heatmap = df_heatmap[df_heatmap['day_of_week'].between(0, 6)]
                
                if not df_heatmap.empty:
                    # Create pivot table
                    heatmap_pivot = df_heatmap.pivot_table(
                        index='day_of_week', 
                        columns='week_number', 
                        values='otif_rate',
                        aggfunc='mean'
                    )
                    
                    # Map day numbers to names
                    day_mapping = {
                        0: 'Sunday',
                        1: 'Monday', 
                        2: 'Tuesday',
                        3: 'Wednesday',
                        4: 'Thursday',
                        5: 'Friday',
                        6: 'Saturday'
                    }
                    
                    heatmap_pivot.index = heatmap_pivot.index.map(day_mapping)
                    
                    # Create heatmap
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=heatmap_pivot.values,
                        x=[f'Week {int(w)}' for w in heatmap_pivot.columns],
                        y=heatmap_pivot.index,
                        colorscale='RdYlGn',
                        zmid=50,  # ✅ CHANGED from 75 to 50 for better color distribution
                        zmin=0,
                        zmax=100,
                        text=np.round(heatmap_pivot.values, 1),
                        texttemplate='%{text}%',
                        textfont={'size': 10},
                        colorbar=dict(title='OTIF %'),
                        hoverongaps=False,
                        hovertemplate='Week: %{x}<br>Day: %{y}<br>OTIF: %{text}%<extra></extra>'
                    ))
                    
                    fig_heatmap.update_layout(
                        title='📅 OTIF Performance Heatmap (By Day of Week & Week Number)',
                        template='plotly_dark',
                        height=450,
                        xaxis_title='Week Number',
                        yaxis_title='Day of Week'
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                else:
                    st.info('ℹ️ No valid heatmap data after filtering.')
                    
            except Exception as e:
                st.warning(f'⚠️ Unable to create heatmap: {str(e)}')
                st.info('💡 Try selecting a longer date range (at least 2-3 weeks).')
        else:
            st.info(f'ℹ️ Insufficient data for heatmap. Found {len(df_heatmap)} data points, need at least 5 across multiple weeks.')

        st.markdown("---")

        # Trend Visualization
        st.subheader("📈 Performance Trends Over Time")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "OTIF Rate Trend", 
                "On-Time Rate Trend",
                "In-Full Rate Trend", 
                "Daily Order Volume"
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # OTIF Rate
        fig.add_trace(go.Scatter(
            x=df_perf["date"], 
            y=df_perf["otif_rate"],
            name="OTIF Rate", 
            line=dict(color="#00CC96", width=2),
            mode='lines+markers',
            marker=dict(size=4),
            hovertemplate='%{x|%Y-%m-%d}<br>OTIF: %{y:.1f}%<extra></extra>'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df_perf["date"], 
            y=df_perf["otif_rate_ma"],
            name="OTIF MA", 
            line=dict(color="#00CC96", dash="dot", width=1.5),
            hovertemplate='%{x|%Y-%m-%d}<br>MA: %{y:.1f}%<extra></extra>'
        ), row=1, col=1)

        # On-Time Rate
        fig.add_trace(go.Scatter(
            x=df_perf["date"], 
            y=df_perf["on_time_rate"],
            name="On-Time Rate", 
            line=dict(color="#636EFA", width=2),
            mode='lines+markers',
            marker=dict(size=4),
            hovertemplate='%{x|%Y-%m-%d}<br>On-Time: %{y:.1f}%<extra></extra>'
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=df_perf["date"], 
            y=df_perf["on_time_rate_ma"],
            name="On-Time MA", 
            line=dict(color="#636EFA", dash="dot", width=1.5),
            hovertemplate='%{x|%Y-%m-%d}<br>MA: %{y:.1f}%<extra></extra>'
        ), row=1, col=2)

        # In-Full Rate
        fig.add_trace(go.Scatter(
            x=df_perf["date"], 
            y=df_perf["in_full_rate"],
            name="In-Full Rate", 
            line=dict(color="#FFA15A", width=2),
            mode='lines+markers',
            marker=dict(size=4),
            hovertemplate='%{x|%Y-%m-%d}<br>In-Full: %{y:.1f}%<extra></extra>'
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=df_perf["date"], 
            y=df_perf["in_full_rate_ma"],
            name="In-Full MA", 
            line=dict(color="#FFA15A", dash="dot", width=1.5),
            hovertemplate='%{x|%Y-%m-%d}<br>MA: %{y:.1f}%<extra></extra>'
        ), row=2, col=1)

        # Daily Orders
        fig.add_trace(go.Bar(
            x=df_perf["date"], 
            y=df_perf["total_orders"],
            name="Daily Orders", 
            marker_color="#AB63FA",
            hovertemplate='%{x|%Y-%m-%d}<br>Orders: %{y:,}<extra></extra>'
        ), row=2, col=2)

        # Add target line at 75%
        for row in [1, 2]:
            for col in [1, 2]:
                if not (row == 2 and col == 2):
                    fig.add_hline(
                        y=75, 
                        line_dash="dash", 
                        line_color="red", 
                        line_width=1,
                        row=row, col=col,
                        annotation_text="Target: 75%",
                        annotation_position="right"
                    )

        fig.update_layout(
            title_text="📊 Daily Performance Dashboard",
            template="plotly_dark",
            height=800,
            showlegend=False,
            hovermode='x unified'
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Rate (%)", range=[0, 100], row=1, col=1)
        fig.update_yaxes(title_text="Rate (%)", range=[0, 100], row=1, col=2)
        fig.update_yaxes(title_text="Rate (%)", range=[0, 100], row=2, col=1)
        fig.update_yaxes(title_text="Orders", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Box Plot - Delivery Time Distribution
        st.subheader("📦 Delivery Time Distribution by Category")
        
        boxplot_sql = """
            SELECT
            COALESCE(p.category, 'Unknown') AS category,
            (f.actual_delivery_date - f.order_placement_date) AS delivery_days,
            CASE WHEN f."On Time" = '1' THEN 'On Time' ELSE 'Late' END AS timing
            FROM fact_order_line f
            LEFT JOIN dim_products p ON f.product_id = p.product_id
            WHERE f.actual_delivery_date BETWEEN :start_date AND :end_date
            AND f.actual_delivery_date > f.order_placement_date
            AND (f.actual_delivery_date - f.order_placement_date) > 0
            AND (f.actual_delivery_date - f.order_placement_date) < 100
            LIMIT 5000;
        """
        df_boxplot = safe_query(boxplot_sql, params)
        
        if not df_boxplot.empty:
            try:
                df_boxplot['delivery_days'] = pd.to_numeric(df_boxplot['delivery_days'], errors='coerce')
                df_boxplot = df_boxplot.dropna(subset=['delivery_days'])
                
                fig_box = px.box(
                    df_boxplot,
                    x='category',
                    y='delivery_days',
                    color='timing',
                    title="📦 Delivery Time Distribution by Product Category",
                    labels={'delivery_days': 'Delivery Days', 'category': 'Product Category'},
                    color_discrete_map={'On Time': '#00CC96', 'Late': '#EF553B'},
                    points="outliers"
                )
                fig_box.update_layout(
                    template="plotly_dark", 
                    height=450, 
                    xaxis_tickangle=-45,
                    hovermode='closest'
                )
                st.plotly_chart(fig_box, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                avg_delivery = df_boxplot['delivery_days'].mean()
                median_delivery = df_boxplot['delivery_days'].median()
                on_time_pct = (len(df_boxplot[df_boxplot['timing'] == 'On Time']) / len(df_boxplot) * 100)
                
                col1.metric("📊 Avg Delivery Time", f"{avg_delivery:.1f} days")
                col2.metric("📊 Median Delivery Time", f"{median_delivery:.0f} days")
                col3.metric("✅ On-Time %", f"{on_time_pct:.1f}%")
                
            except Exception as e:
                st.warning(f"⚠️ Unable to create box plot: {str(e)}")
        else:
            st.info("ℹ️ No delivery time data available for selected date range.")
        
        st.markdown("---")
        
        # Product Performance Analysis
        st.subheader("📦 Top Product Performance")
        
        prod_sql = f"""
            SELECT
                COALESCE(p.product_name, 'Unknown Product') AS product_name,
                COALESCE(p.category, 'Unknown') AS category,
                COUNT(*) AS total_orders,
                SUM(f.order_qty) AS total_qty_ordered,
                SUM(f.delivery_qty) AS total_qty_delivered,
                {yes_no_rate('SUM(CASE WHEN f."On Time In Full" = \'1\' THEN 1 ELSE 0 END)')} AS otif_rate,
                {yes_no_rate('SUM(CASE WHEN f."On Time" = \'1\' THEN 1 ELSE 0 END)')} AS on_time_rate,
                {yes_no_rate('SUM(CASE WHEN f."In Full" = \'1\' THEN 1 ELSE 0 END)')} AS in_full_rate
            FROM fact_order_line f
            LEFT JOIN dim_products p ON f.product_id = p.product_id
            WHERE f.actual_delivery_date BETWEEN :start_date AND :end_date
            GROUP BY p.product_name, p.category
            HAVING COUNT(*) >= 5
            ORDER BY otif_rate DESC
            LIMIT 25;
        """
        df_prod = safe_query(prod_sql, params)
        
        if not df_prod.empty:
            try:
                # Convert to numeric
                for col in ['total_orders', 'total_qty_ordered', 'total_qty_delivered', 'otif_rate', 'on_time_rate', 'in_full_rate']:
                    df_prod[col] = pd.to_numeric(df_prod[col], errors='coerce').fillna(0)
                
                # Top 15 products chart
                top_15 = df_prod.head(15)
                
                fig_prod = go.Figure()
                
                fig_prod.add_trace(go.Bar(
                    x=top_15['product_name'],
                    y=top_15['otif_rate'],
                    marker=dict(
                        color=top_15['otif_rate'],
                        colorscale='RdYlGn',
                        cmin=0,
                        cmax=100,
                        colorbar=dict(title="OTIF %")
                    ),
                    text=[f"{v:.1f}%" for v in top_15['otif_rate']],
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>OTIF Rate: %{y:.1f}%<br>Orders: %{customdata}<extra></extra>',
                    customdata=top_15['total_orders']
                ))
                
                fig_prod.update_layout(
                    title="🎯 Top 15 Products by OTIF Rate",
                    template="plotly_dark",
                    xaxis_tickangle=-45,
                    height=500,
                    yaxis=dict(title="OTIF Rate (%)", range=[0, 105]),
                    xaxis_title="Product Name"
                )
                st.plotly_chart(fig_prod, use_container_width=True)
                
                # Data table
                st.dataframe(
                    df_prod.head(25).style.format({
                        'total_orders': '{:,.0f}',
                        'total_qty_ordered': '{:,.0f}',
                        'total_qty_delivered': '{:,.0f}',
                        'otif_rate': '{:.1f}%',
                        'on_time_rate': '{:.1f}%',
                        'in_full_rate': '{:.1f}%'
                    }).background_gradient(
                        subset=['otif_rate', 'on_time_rate', 'in_full_rate'], 
                        cmap='RdYlGn',
                        vmin=0,
                        vmax=100
                    ),
                    use_container_width=True,
                    height=400
                )
            except Exception as e:
                st.warning(f"⚠️ Unable to create product performance visualization: {str(e)}")
                st.dataframe(df_prod, use_container_width=True)
        else:
            st.info("ℹ️ No product performance data available (need products with at least 5 orders).")
        
        st.markdown("---")
        
        # Sankey Diagram
        st.subheader("🌊 Order Flow: Category → Delivery Status")
        
        sankey_sql = """
            SELECT
                COALESCE(p.category, 'Unknown') AS category,
                CASE 
                    WHEN f."On Time In Full" = '1' THEN 'Delivered Successfully'
                    WHEN f."On Time" = '0' THEN 'Late Delivery'
                    WHEN f."In Full" = '0' THEN 'Partial Delivery'
                    ELSE 'Other Issues'
                END AS delivery_status,
                COUNT(*) AS order_count
            FROM fact_order_line f
            LEFT JOIN dim_products p ON f.product_id = p.product_id
            WHERE f.actual_delivery_date BETWEEN :start_date AND :end_date
            GROUP BY p.category, delivery_status
            HAVING COUNT(*) >= 3
            ORDER BY order_count DESC;
        """
        df_sankey = safe_query(sankey_sql, params)
        
        if not df_sankey.empty and len(df_sankey) >= 3:
            try:
                df_sankey['order_count'] = pd.to_numeric(df_sankey['order_count'], errors='coerce').fillna(0).astype(int)
                
                all_categories = df_sankey['category'].unique().tolist()
                all_statuses = df_sankey['delivery_status'].unique().tolist()
                node_labels = all_categories + all_statuses
                
                source = []
                target = []
                value = []
                
                for _, row in df_sankey.iterrows():
                    source.append(node_labels.index(row['category']))
                    target.append(node_labels.index(row['delivery_status']))
                    value.append(int(row['order_count']))
                
                # Color nodes
                category_colors = ['#636EFA'] * len(all_categories)
                status_colors = []
                for s in all_statuses:
                    if 'Success' in s:
                        status_colors.append('#00CC96')
                    elif 'Late' in s:
                        status_colors.append('#EF553B')
                    elif 'Partial' in s:
                        status_colors.append('#FFA15A')
                    else:
                        status_colors.append('#AB63FA')
                
                node_colors = category_colors + status_colors
                
                fig_sankey = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=20,
                        thickness=25,
                        label=node_labels,
                        color=node_colors,
                        hovertemplate='%{label}<br>Total Orders: %{value:,}<extra></extra>'
                    ),
                    link=dict(
                        source=source,
                        target=target,
                        value=value,
                        color='rgba(100,100,100,0.3)',
                        hovertemplate='%{source.label} → %{target.label}<br>Orders: %{value:,}<extra></extra>'
                    )
                )])
                
                fig_sankey.update_layout(
                    title="🌊 Order Flow: Product Category → Delivery Outcome",
                    template="plotly_dark",
                    height=550,
                    font=dict(size=12)
                )
                st.plotly_chart(fig_sankey, use_container_width=True)
                
                # Summary table
                st.subheader("📋 Order Flow Summary")
                summary = df_sankey.pivot_table(
                    index='category',
                    columns='delivery_status',
                    values='order_count',
                    aggfunc='sum',
                    fill_value=0
                ).astype(int)
                
                st.dataframe(
                    summary.style.format("{:,}").background_gradient(cmap='YlOrRd', axis=1),
                    use_container_width=True
                )
                
            except Exception as e:
                st.warning(f"⚠️ Unable to create Sankey diagram: {str(e)}")
        else:
            st.info("ℹ️ Insufficient data for order flow analysis (need at least 3 category-status combinations).")

# -----------------------------------------------------------------------------
#  TAB 6 — ALERTS
# -----------------------------------------------------------------------------
with tabs[5]:
    st.header("6️⃣ Alerts Monitoring")

    alerts_recent_sql = """
        SELECT 
        DATE_TRUNC('hour', actual_delivery_date) AS hour,
        COUNT(*) AS total_orders,
        SUM(CASE WHEN "On Time" = '0' THEN 1 ELSE 0 END) AS late_orders,
        ROUND(SUM(CASE WHEN "On Time" = '0' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS late_rate
      FROM fact_order_line
      WHERE actual_delivery_date BETWEEN :start_date AND :end_date
      GROUP BY 1
      ORDER BY hour DESC
      LIMIT 100;
    """
    df_alerts = safe_query(alerts_recent_sql, params)
    
    if df_alerts.empty:
        st.success("✅ No late orders detected in selected period.")
    else:
        df_alerts['hour'] = pd.to_datetime(df_alerts['hour'])
        
        total_late = int(df_alerts['late_orders'].sum())
        total_orders = int(df_alerts['total_orders'].sum())
        late_pct = (total_late / total_orders * 100) if total_orders > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🚨 Late Orders", f"{total_late:,}")
        col2.metric("📊 Total Orders", f"{total_orders:,}")
        col3.metric("⚠️ Late Rate", f"{late_pct:.1f}%")
        
        st.markdown("---")
        st.bar_chart(df_alerts.set_index("hour")[["late_orders"]])


# ----------------------------------------------------------------------------- 
#  TAB 7 — TRENDS (WITH CORRELATION HEATMAP & 3D SCATTER) - CORRECTED VERSION
# ----------------------------------------------------------------------------- 
with tabs[6]:
    st.header("7️⃣ Performance Trends & Analytics")

    # CORRECTED: Weekly Trends SQL - Fixed for '1'/'0' data format
    trends_sql = f"""
        SELECT
            DATE_TRUNC('week', actual_delivery_date) AS week,
            COUNT(*) AS total_orders,
            ROUND(SUM(CASE WHEN "On Time" = '1' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS weekly_on_time,
            ROUND(SUM(CASE WHEN "In Full" = '1' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS weekly_in_full,
            ROUND(SUM(CASE WHEN "On Time In Full" = '1' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS weekly_otif,
            AVG(order_qty) AS avg_order_size
        FROM fact_order_line
        WHERE actual_delivery_date BETWEEN :start_date AND :end_date
        GROUP BY 1
        ORDER BY week;
    """
    
    df_trends = safe_query(trends_sql, params)

    if df_trends.empty:
        st.warning("No weekly trend data available for selected period.")
    else:
        df_trends["week"] = pd.to_datetime(df_trends["week"])

        # Summary metrics - NOW WITH REAL VALUES (not 0%)
        col1, col2, col3, col4 = st.columns(4)
        avg_weekly = safe_float(df_trends['total_orders'].mean())
        avg_otif = safe_float(df_trends['weekly_otif'].mean())
        best_otif = safe_float(df_trends['weekly_otif'].max())
        worst_otif = safe_float(df_trends['weekly_otif'].min())

        col1.metric("📊 Avg Weekly Orders", f"{avg_weekly:.0f}")
        col2.metric("🎯 Avg OTIF", f"{avg_otif:.1f}%")  # Will show ~51% instead of 0%
        col3.metric("🏆 Best Week", f"{best_otif:.1f}%")
        col4.metric("⚠️ Worst Week", f"{worst_otif:.1f}%")

        st.markdown("---")

        # Correlation Heatmap - NOW WITH REAL DATA
        st.subheader("🔗 Metric Correlation Analysis")
        if len(df_trends) >= 3:
            corr_cols = ['total_orders', 'weekly_on_time', 'weekly_in_full', 'weekly_otif', 'avg_order_size']
            corr_matrix = df_trends[corr_cols].corr()

            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=['Order Volume', 'On-Time %', 'In-Full %', 'OTIF %', 'Avg Order Size'],
                y=['Order Volume', 'On-Time %', 'In-Full %', 'OTIF %', 'Avg Order Size'],
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 11},
                colorbar=dict(title="Correlation")
            ))

            fig_corr.update_layout(
                title="🔗 Performance Metrics Correlation Matrix",
                template="plotly_dark",
                height=500,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("---")

        # Weekly Trends Line Chart - NOW WITH REAL VALUES
        st.subheader("📈 Weekly Performance Trends")
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=df_trends["week"], y=df_trends["weekly_otif"],
            name="OTIF %", line=dict(color="green", width=3),
            mode='lines+markers'
        ))
        fig_trend.add_trace(go.Scatter(
            x=df_trends["week"], y=df_trends["weekly_on_time"],
            name="On-Time %", line=dict(color="blue", width=2),
            mode='lines+markers'
        ))
        fig_trend.add_trace(go.Scatter(
            x=df_trends["week"], y=df_trends["weekly_in_full"],
            name="In-Full %", line=dict(color="orange", width=2),
            mode='lines+markers'
        ))
        
        # Add target line at 75%
        fig_trend.add_hline(y=75, line_dash="dash", line_color="red", 
                          annotation_text="Target: 75%", annotation_position="right")
        
        fig_trend.update_layout(
            title="📈 Weekly Performance Trends",
            template="plotly_dark",
            xaxis_title="Week",
            yaxis_title="Performance Rate (%)",
            height=520,
            hovermode='x unified',
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("---")

        # CORRECTED: 3D Scatter Plot SQL - Fixed for '1'/'0' format
        st.subheader("🎯 3D Product Performance Analysis")
        
        scatter_3d_sql = f"""
            SELECT
                p.product_name,
                p.category,
                AVG(f.order_qty) AS avg_order_qty,
                AVG((f.actual_delivery_date - f.order_placement_date)) AS avg_delivery_days,
                ROUND(SUM(CASE WHEN f."On Time In Full" = '1' THEN 1 ELSE 0 END) * 100.0 / NULLIF(COUNT(*), 0), 2) AS otif_rate,
                COUNT(*) AS total_orders
            FROM fact_order_line f
            JOIN dim_products p ON f.product_id = p.product_id
            WHERE f.actual_delivery_date BETWEEN :start_date AND :end_date
            AND f.actual_delivery_date > f.order_placement_date
            GROUP BY p.product_name, p.category
            HAVING COUNT(*) >= 10
            LIMIT 50;
        """
        
        df_3d = safe_query(scatter_3d_sql, params)

        if not df_3d.empty and len(df_3d) >= 3:
            fig_3d = px.scatter_3d(
                df_3d,
                x='avg_order_qty',
                y='avg_delivery_days',
                z='otif_rate',
                color='category',
                size='total_orders',
                hover_name='product_name',
                title="🎯 3D Analysis: Order Qty vs Delivery Time vs OTIF",
                labels={
                    'avg_order_qty': 'Avg Order Quantity',
                    'avg_delivery_days': 'Avg Delivery Days',
                    'otif_rate': 'OTIF Rate (%)'
                }
            )

            fig_3d.update_layout(template="plotly_dark", height=700)
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("⚠️ Insufficient data for 3D analysis (need at least 3 products with 10+ orders)")

# -----------------------------------------------------------------------------
#  TAB 8 — CONFIGURATION
# -----------------------------------------------------------------------------
with tabs[7]:
    st.header("8️⃣ Configuration Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        otif_thresh = st.slider("OTIF Threshold", 50, 95, 75)
        infull_thresh = st.slider("In-Full Threshold", 50, 98, 80)
        ontime_thresh = st.slider("On-Time Threshold", 50, 95, 75)
        st.info(
            f"Current thresholds: OTIF < {otif_thresh}%, "
            f"In-Full < {infull_thresh}%, "
            f"On-Time < {ontime_thresh}%."
        )
    
    with col2:
        st.subheader("📊 System Information")
        st.text(f"Database: Connected ✅")
        st.text(f"Date Range: {(end_date - start_date).days} days")
        st.text(f"Refresh Counter: {st.session_state.refresh_counter}")


# -----------------------------------------------------------------------------
#  TAB 9 — EXECUTIVE SUMMARY (WITH GAUGE CHARTS & WATERFALL)
# -----------------------------------------------------------------------------
with tabs[8]:
    st.header("9️⃣ Executive Summary")

    summary_sql = f"""
        SELECT
            COUNT(*) AS total_orders,
            COUNT(DISTINCT customer_id) AS unique_customers,
            COALESCE(SUM(CASE WHEN on_time = 1 THEN 1 ELSE 0 END), 0) AS on_time_orders,
            COALESCE(SUM(CASE WHEN in_full = '1' THEN 1 ELSE 0 END), 0) AS in_full_orders,
            COALESCE(SUM(CASE WHEN otif = '1' THEN 1 ELSE 0 END), 0) AS otif_orders,
            {yes_no_rate("SUM(CASE WHEN on_time = 1 THEN 1 ELSE 0 END)")} AS on_time_percentage,
            {yes_no_rate("SUM(CASE WHEN in_full = '1' THEN 1 ELSE 0 END)")} AS in_full_percentage,
            {yes_no_rate("SUM(CASE WHEN otif = '1' THEN 1 ELSE 0 END)")} AS otif_percentage
        FROM fact_aggregate
        WHERE order_placement_date BETWEEN :start_date AND :end_date;
    """

    df_summary = safe_query(summary_sql, params)
    
    if df_summary.empty:
        st.warning("No summary data found for selected period.")
    else:
        total_orders = int(df_summary.at[0, "total_orders"])
        unique_customers = int(df_summary.at[0, "unique_customers"])
        on_time = safe_float(df_summary.at[0, "on_time_percentage"])
        in_full = safe_float(df_summary.at[0, "in_full_percentage"])
        otif = safe_float(df_summary.at[0, "otif_percentage"])

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("📦 Total Orders", f"{total_orders:,}")
        col2.metric("👥 Customers", f"{unique_customers:,}")
        col3.metric("⏰ On-Time", f"{on_time:.1f}%")
        col4.metric("📦 In-Full", f"{in_full:.1f}%")
        col5.metric("🎯 OTIF", f"{otif:.1f}%")

        st.markdown("---")
        
        # ⭐ NEW: Gauge Charts
        st.subheader("🎯 KPI Performance Gauges")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_gauge_otif = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=otif,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "OTIF Rate", 'font': {'size': 20}},
                delta={'reference': 75, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': '#EF553B'},
                        {'range': [50, 75], 'color': '#FFA500'},
                        {'range': [75, 100], 'color': '#00CC96'}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 75}
                }
            ))
            fig_gauge_otif.update_layout(template="plotly_dark", height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge_otif, use_container_width=True)
        
        with col2:
            fig_gauge_on_time = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=on_time,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "On-Time Rate", 'font': {'size': 20}},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 60], 'color': '#EF553B'},
                        {'range': [60, 80], 'color': '#FFA500'},
                        {'range': [80, 100], 'color': '#00CC96'}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 80}
                }
            ))
            fig_gauge_on_time.update_layout(template="plotly_dark", height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge_on_time, use_container_width=True)
        
        with col3:
            fig_gauge_in_full = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=in_full,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "In-Full Rate", 'font': {'size': 20}},
                delta={'reference': 85},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkorange"},
                    'steps': [
                        {'range': [0, 70], 'color': '#EF553B'},
                        {'range': [70, 85], 'color': '#FFA500'},
                        {'range': [85, 100], 'color': '#00CC96'}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 85}
                }
            ))
            fig_gauge_in_full.update_layout(template="plotly_dark", height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge_in_full, use_container_width=True)
        
        st.markdown("---")
        
        # ⭐ NEW: Waterfall Chart
        st.subheader("💧 Order Performance Waterfall")
        
        on_time_orders = int(df_summary.at[0, "on_time_orders"])
        in_full_orders = int(df_summary.at[0, "in_full_orders"])
        otif_orders = int(df_summary.at[0, "otif_orders"])
        
        fig_waterfall = go.Figure(go.Waterfall(
            name="Performance",
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Total Orders", "On-Time Orders", "In-Full Orders", "OTIF Orders"],
            y=[total_orders, on_time_orders - total_orders, in_full_orders - on_time_orders, otif_orders],
            text=[f"{total_orders:,}", f"{on_time_orders:,}", f"{in_full_orders:,}", f"{otif_orders:,}"],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#EF553B"}},
            increasing={"marker": {"color": "#00CC96"}},
            totals={"marker": {"color": "#636EFA"}}
        ))
        
        fig_waterfall.update_layout(
            title="💧 Order Performance Breakdown",
            template="plotly_dark",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        st.markdown("---")

        col1, col2 = st.columns(2)
        
        with col1:
            bar_df = pd.DataFrame({
                "Metric": ["On-Time", "In-Full", "OTIF"],
                "Percentage": [on_time, in_full, otif],
            })

            fig_bar = px.bar(
                bar_df,
                x="Metric",
                y="Percentage",
                color="Metric",
                text="Percentage",
                title="📊 Executive KPIs",
                labels={"Percentage": "Percentage (%)"},
                color_discrete_map={"On-Time": "#1f77b4", "In-Full": "#2ca02c", "OTIF": "#ff7f0e"},
            )
            fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_bar.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            pie_df = pd.DataFrame({
                "Category": ["On-Time Orders", "In-Full Orders", "OTIF Orders"],
                "Value": [on_time_orders, in_full_orders, otif_orders],
            })
            fig_pie = px.pie(
                pie_df,
                names="Category",
                values="Value",
                title="🥧 Orders Distribution by KPI",
                color_discrete_sequence=px.colors.sequential.RdBu,
            )
            fig_pie.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("---")
        if otif >= 85:
            st.success(f"🎉 Excellent! OTIF at {otif:.1f}% - exceeding target")
        elif otif >= 75:
            st.info(f"✅ Good! OTIF at {otif:.1f}% - meeting target")
        else:
            st.error(f"⚠️ Action needed! OTIF at {otif:.1f}% - below target of 75%")


# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #888;'>
        <p>Supply Chain Analytics Dashboard | Period: {start_date} to {end_date}</p>
        <p>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Refresh: {st.session_state.refresh_counter}</p>
    </div>
    """,
    unsafe_allow_html=True
)