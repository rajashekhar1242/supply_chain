import os
import smtplib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
from sqlalchemy import create_engine, text
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from dotenv import load_dotenv
import plotly.express as px
import traceback
import html 
# ------------------ 1. Load environment variables ------------------
load_dotenv()
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
ALERT_FROM = os.getenv("ALERT_FROM", SMTP_USER)
ALERT_TO = os.getenv("ALERT_TO", "")
DATABASE_URL = os.getenv("DATABASE_URL")

# ------------------ 2. Initialize database engine ------------------
engine = create_engine(DATABASE_URL) if DATABASE_URL else None

# ------------------ 3. Configure logging (simplified) ------------------
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "alert_worker.log"

# Reset existing handlers
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",   # simple format
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("alert_worker")

# ------------------ 4. Queries & Fetchers ---------------
INVENTORY_QUERY = """
    SELECT di.product_id, di.item_name, di.stock_level,
           COALESCE(pp.reorder_point, 0) AS reorder_point,
           COALESCE(pp.overstock_threshold, 500) AS overstock_threshold
    FROM derived_inventory di
    LEFT JOIN policy_parameters pp ON pp.product_id = di.product_id
"""

def fetch_risk_data() -> pd.DataFrame:
    if engine is None:
        logger.error("No DATABASE_URL set - cannot fetch risk data.")
        return pd.DataFrame()
    try:
        return pd.read_sql("""
            SELECT p.customer_id, c.customer_name, p.risk_score, p.risk_label
            FROM kpi_risk_predictions p
            JOIN dim_customers c ON p.customer_id = c.customer_id;
        """, engine)
    except Exception as e:
        logger.error(f"Failed to fetch risk data: {e}\n{traceback.format_exc()}")
        return pd.DataFrame()

def fetch_executive_summary() -> dict:
    if engine is None:
        logger.error("No DATABASE_URL set - cannot fetch executive summary.")
        return {}
    try:
        query = """
        SELECT 
            COUNT(*) AS total_orders,
            ROUND(SUM(CASE WHEN "On Time"='1' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),2) AS on_time_pct,
            ROUND(SUM(CASE WHEN "In Full"='1' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),2) AS in_full_pct,
            ROUND(SUM(CASE WHEN "On Time In Full"='1' THEN 1 ELSE 0 END) * 100.0 / COUNT(*),2) AS otif_pct
        FROM fact_order_line
        WHERE actual_delivery_date IS NOT NULL;
        """
        df = pd.read_sql(query, engine)
        return df.iloc[0].to_dict() if not df.empty else {}
    except Exception as e:
        logger.error(f"Failed to fetch Executive Summary: {e}\n{traceback.format_exc()}")
        return {}

# ------------------ 5. Alert checks ------------------
def check_inventory() -> List[Tuple[str, str, str, str]]:
    alerts = []
    if engine is None:
        logger.error("No DATABASE_URL set - cannot check inventory.")
        return alerts
    try:
        df = pd.read_sql(INVENTORY_QUERY, engine)
        for _, row in df.iterrows():
            stock = float(row["stock_level"] or 0)
            if stock <= float(row["reorder_point"] or 0):
                alerts.append(("STOCKOUT", "Product", row["item_name"], f"STOCKOUT: {row['item_name']} (stock={stock})"))
            elif stock >= float(row["overstock_threshold"] or 500):
                alerts.append(("OVERSTOCK", "Product", row["item_name"], f"OVERSTOCK: {row['item_name']} (stock={stock})"))
    except Exception as e:
        logger.error(f"Inventory check failed: {e}\n{traceback.format_exc()}")
    return alerts

def check_kpis() -> List[Tuple[str, str, str, str]]:
    alerts = []
    if engine is None:
        logger.error("No DATABASE_URL set - cannot check KPIs.")
        return alerts
    try:
        df = pd.read_sql("SELECT * FROM kpi_summary;", engine)
        for _, row in df.iterrows():
            cname = row.get("customer_name", "Unknown")
            if row.get("otif_pct", 100) < row.get("target_otif", 100):
                alerts.append(("KPI", "Customer", cname, f"OTIF below target: {row.get('otif_pct',0)}% vs {row.get('target_otif',0)}%"))
            if row.get("on_time_pct", 100) < row.get("target_on_time", 100):
                alerts.append(("KPI", "Customer", cname, f"On-Time below target: {row.get('on_time_pct',0)}% vs {row.get('target_on_time',0)}%"))
            if row.get("in_full_pct", 100) < row.get("target_in_full", 100):
                alerts.append(("KPI", "Customer", cname, f"In-Full below target: {row.get('in_full_pct',0)}% vs {row.get('target_in_full',0)}%"))
    except Exception as e:
        logger.error(f"KPI check failed: {e}\n{traceback.format_exc()}")
    return alerts

def check_risk_alerts(top_n: int = 5) -> Tuple[List[Tuple[str, str, str, str]], pd.DataFrame]:
    alerts = []
    df = fetch_risk_data()
    if df.empty:
        return alerts, pd.DataFrame()
    try:
        df['risk_score'] = pd.to_numeric(df['risk_score'], errors='coerce')
        df = df.dropna(subset=['risk_score'])

        for _, row in df.iterrows():
            score = row['risk_score']
            customer = row.get("customer_name", "Unknown")
            label = row.get("risk_label", "").capitalize()

            # Dynamic severity levels based on score thresholds
            if score > 0.9:
                level = "Critical"
            elif score > 0.8:
                level = "High"
            elif score > 0.6:
                level = "Moderate"
            else:
                level = "Low"

            # Only trigger alerts for moderate and above
            if score > 0.6:
                alerts.append((
                    "RISK",
                    "Customer",
                    customer,
                    f"{level} Risk Detected — Probability Score: {score:.2f} ({label})"
                ))

        # Select top N risky customers for summary table
        top_risky = df.nlargest(top_n, 'risk_score')
        return alerts, top_risky

    except Exception as e:
        logger.error(f"Risk check failed: {e}\n{traceback.format_exc()}")
        return alerts, pd.DataFrame()


# ------------------ 6. Store Alerts ------------------
def store_alert(alert_type: str, entity_type: str, entity_name: str, detail: str):
    if engine is None:
        logger.error("No DATABASE_URL set - cannot store alert.")
        return
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""INSERT INTO alert_log (alert_time, alert_type, entity_type, entity_name, detail)
                        VALUES (:alert_time, :alert_type, :entity_type, :entity_name, :detail)"""),
                {"alert_time": datetime.now(timezone.utc),
                 "alert_type": alert_type,
                 "entity_type": entity_type,
                 "entity_name": entity_name,
                 "detail": detail}
            )
    except Exception as e:
        logger.error(f"Failed to store alert: {e}\n{traceback.format_exc()}")

# ------------------ 7. Graphs (return PNG bytes) ------------------
def plot_kpi_summary_png(exec_summary: dict) -> Optional[bytes]:
    """
    Build a small KPI bar chart from exec_summary dict and return PNG bytes.
    """
    if not exec_summary:
        return None
    try:
        df = pd.DataFrame({
            "KPI": ["OTIF", "On-Time", "In-Full"],
            "Actual": [
                float(exec_summary.get("otif_pct") or 0),
                float(exec_summary.get("on_time_pct") or 0),
                float(exec_summary.get("in_full_pct") or 0),
            ],
            "Target": [95.0, 95.0, 95.0]
        })
        fig = px.bar(df, x="KPI", y=["Actual", "Target"], barmode="group", text_auto=True,
                     title="KPI vs Target")
        img_bytes = fig.to_image(format="png")
        return img_bytes
    except Exception as e:
        logger.error(f"KPI graph failed: {e}\n{traceback.format_exc()}")
        return None

def plot_inventory_png() -> Optional[bytes]:
    """
    Read inventory, compute status and render a stock-level bar chart, return PNG bytes.
    """
    if engine is None:
        logger.error("No DATABASE_URL set - cannot plot inventory.")
        return None
    try:
        df = pd.read_sql(INVENTORY_QUERY, engine)
        if df.empty:
            return None
        df['status'] = df.apply(
            lambda r: 'STOCKOUT' if float(r['stock_level'] or 0) <= float(r['reorder_point'] or 0)
            else ('OVERSTOCK' if float(r['stock_level'] or 0) >= float(r['overstock_threshold'] or 500) else 'NORMAL'),
            axis=1
        )
        # Simple bar chart showing stock level by product
        fig = px.bar(df, x='item_name', y='stock_level', color='status', text='stock_level',
                     title='Inventory Stock Levels')
        fig.update_layout(xaxis_title='Product', yaxis_title='Stock Level', showlegend=True)
        img_bytes = fig.to_image(format="png")
        return img_bytes
    except Exception as e:
        logger.error(f"Inventory graph failed: {e}\n{traceback.format_exc()}")
        return None

# ------------------ 8. Format Email ------------------
def format_email_body(all_alerts: List[Tuple[str,str,str,str]], top_risky: pd.DataFrame, exec_summary: dict) -> Tuple[str, str]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    alert_count = len(all_alerts)

    inventory_alerts = [a for a in all_alerts if a[0] in ["STOCKOUT","OVERSTOCK"]]
    kpi_alerts = [a for a in all_alerts if a[0] == "KPI"]
    risk_alerts = [a for a in all_alerts if a[0] == "RISK"]

    # HTML body - reference images via cid:img1 and cid:img2
    html_body = f"<html><body><h1>🚨 Supply Chain Alert Report</h1><p>Date: {today} | Alerts: {alert_count}</p>"

    if exec_summary:
        html_body += (
            "<h2>Executive Summary</h2>"
            "<table border='1' cellpadding='4' cellspacing='0'>"
            "<tr><th>Total Orders</th><th>OTIF</th><th>On-Time</th><th>In-Full</th></tr>"
        )
        html_body += f"<tr><td>{exec_summary.get('total_orders',0)}</td><td>{exec_summary.get('otif_pct',0)}%</td><td>{exec_summary.get('on_time_pct',0)}%</td><td>{exec_summary.get('in_full_pct',0)}%</td></tr></table>"
        html_body += "<p><img src='cid:img1' alt='KPI Chart'></p>"

    html_body += "<h2>Inventory Alerts</h2>"
    html_body += "<ul>" + "".join(f"<li>{html.escape(a[3])}</li>" for a in inventory_alerts) + "</ul>" if inventory_alerts else "<p>No inventory alerts</p>"
    html_body += "<p><img src='cid:img2' alt='Inventory Chart'></p>"

    html_body += "<h2>KPI Alerts</h2>"
    html_body += "<ul>" + "".join(f"<li>{html.escape(a[3])}</li>" for a in kpi_alerts) + "</ul>" if kpi_alerts else "<p>No KPI alerts</p>"

    html_body += "<h2>Risk Alerts</h2>"
    html_body += "<ul>" + "".join(f"<li>{html.escape(a[3])}</li>" for a in risk_alerts) + "</ul>" if risk_alerts else "<p>No Risk alerts</p>"

    if top_risky is not None and not top_risky.empty:
        html_body += "<h3>Top Risky Customers</h3><table border='1' cellpadding='4' cellspacing='0'><tr><th>Customer</th><th>Risk Score</th><th>Label</th></tr>"
        for _, row in top_risky.iterrows():
            html_body += f"<tr><td>{html.escape(str(row.get('customer_name','')))}</td><td>{row.get('risk_score',0):.3f}</td><td>{html.escape(str(row.get('risk_label','')))}</td></tr>"
        html_body += "</table>"

    html_body += "</body></html>"

    # Plain text fallback
    text_lines = [
        "Supply Chain Alert Report",
        f"Date: {today}",
        f"Total Alerts: {alert_count}",
        ""
    ]
    if exec_summary:
        text_lines += [
            "Executive Summary:",
            f"Total Orders: {exec_summary.get('total_orders',0)}",
            f"OTIF: {exec_summary.get('otif_pct',0)}%",
            f"On-Time: {exec_summary.get('on_time_pct',0)}%",
            f"In-Full: {exec_summary.get('in_full_pct',0)}%",
            ""
        ]
    text_lines += ["Inventory Alerts:"] + ([a[3] for a in inventory_alerts] if inventory_alerts else ["No inventory alerts"]) + [""]
    text_lines += ["KPI Alerts:"] + ([a[3] for a in kpi_alerts] if kpi_alerts else ["No KPI alerts"]) + [""]
    text_lines += ["Risk Alerts:"] + ([a[3] for a in risk_alerts] if risk_alerts else ["No Risk alerts"]) + [""]

    if top_risky is not None and not top_risky.empty:
        text_lines += ["Top Risky Customers:"]
        text_lines += [f"{r['customer_name']} | {r['risk_score']:.3f} | {r['risk_label']}" for _, r in top_risky.iterrows()]

    text_body = "\n".join(text_lines)

    return text_body, html_body

# ------------------ 9. Send Email (with attachments as inline images) ------------------
def send_email(subject: str, html_body: str, text_body: str = "", debug: bool=False, attachments: List[tuple]=None):
    if not SMTP_USER or not SMTP_PASS or not ALERT_TO:
        logger.warning("Email not configured properly. Skipping send.")
        return
    recipients = [r.strip() for r in ALERT_TO.split(",") if r.strip()]
    if not recipients:
        logger.warning("No valid recipients.")
        return
    try:
        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"] = ALERT_FROM
        msg["To"] = ", ".join(recipients)

        alt = MIMEMultipart("alternative")
        if text_body:
            alt.attach(MIMEText(text_body, "plain", "utf-8"))
        alt.attach(MIMEText(html_body, "html", "utf-8"))
        msg.attach(alt)

        # Attach inline images: attachments is list of (filename, img_bytes)
        if attachments:
            for idx, (filename, img_bytes) in enumerate(attachments, start=1):
                try:
                    img = MIMEImage(img_bytes, name=filename)
                    img.add_header("Content-ID", f"<img{idx}>")
                    img.add_header("Content-Disposition", "inline", filename=filename)
                    msg.attach(img)
                except Exception:
                    logger.error(f"Failed to attach image {filename}: {traceback.format_exc()}")

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.set_debuglevel(1 if debug else 0)
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(ALERT_FROM, recipients, msg.as_string())

        logger.info(f"Email sent successfully to {', '.join(recipients)}")
    except Exception as e:
        logger.error(f"Email sending failed: {e}\n{traceback.format_exc()}")

# ------------------ 10. Run Alerts ------------------
def run_alert_checks() -> dict:
    all_alerts = []
    try:
        all_alerts.extend(check_inventory())
        all_alerts.extend(check_kpis())
        risk_alerts, top_risky = check_risk_alerts()
        all_alerts.extend(risk_alerts)
        exec_summary = fetch_executive_summary()

        # Store alerts in DB
        for alert in all_alerts:
            store_alert(*alert)

        # Prepare email body
        text_body, html_body = format_email_body(all_alerts, top_risky, exec_summary)

        # Prepare graph attachments as PNG bytes
        attachments = []
        kpi_img = plot_kpi_summary_png(exec_summary)
        inv_img = plot_inventory_png()
        if kpi_img:
            attachments.append(("kpi.png", kpi_img))
        if inv_img:
            attachments.append(("inventory.png", inv_img))

        # Send email
        subject = f"[SupplyChain Alerts {datetime.now(timezone.utc).strftime('%Y-%m-%d')}]"
        send_email(subject, html_body, text_body, debug=True, attachments=attachments)

        logger.info(f"Processed {len(all_alerts)} alerts.")
        return {"alerts_generated": len(all_alerts)}
    except Exception as e:
        logger.error(f"Alert check failed: {e}\n{traceback.format_exc()}")
        return {"alerts_generated": 0}

if __name__ == "__main__":
    run_alert_checks()
