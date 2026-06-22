from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

# -------------------------------
# Import Engines
# -------------------------------
from ingestion.main import run_ingestion

from analytics.demand_engine import run as run_demand_features
from analytics.demand_forecasting import run as run_forecast
from analytics.inventory_optimization_engine import run as run_inventory

from analytics.customer_risk import run as run_customer_risk
from analytics.multi_warehouse_optimization import run as run_multi_warehouse
from analytics.network_optimization_engine import run as run_network


# -------------------------------
# Default Args
# -------------------------------
default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
    "depends_on_past": False,
}


with DAG(
    dag_id="supplychain_enterprise_pipeline",
    description="Structured ML Supply Chain Orchestration",
    default_args=default_args,
    start_date=days_ago(1),
    schedule_interval="*/30 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["enterprise", "supply_chain"],
) as dag:

    start = EmptyOperator(task_id="start")

    # -------------------------------
    # 1️⃣ Ingestion
    # -------------------------------
    ingestion = PythonOperator(
        task_id="data_ingestion",
        python_callable=run_ingestion,
        op_kwargs={"env": "default", "once": True},
    )

    # -------------------------------
    # 2️⃣ Forecasting Chain
    # -------------------------------
    demand_features = PythonOperator(
        task_id="demand_features",
        python_callable=run_demand_features,
    )

    demand_forecasting = PythonOperator(
        task_id="demand_forecasting",
        python_callable=run_forecast,
    )

    inventory_optimization = PythonOperator(
        task_id="inventory_optimization",
        python_callable=run_inventory,
    )

    # -------------------------------
    # 3️⃣ Parallel Independent Tasks
    # -------------------------------
    customer_risk = PythonOperator(
        task_id="customer_risk",
        python_callable=run_customer_risk,
    )

    multi_warehouse = PythonOperator(
        task_id="multi_warehouse_optimization",
        python_callable=run_multi_warehouse,
    )

    network_optimization = PythonOperator(
        task_id="network_optimization",
        python_callable=run_network,
    )

    # -------------------------------
    # 4️⃣ Join All Branches
    # -------------------------------
    all_done = EmptyOperator(
        task_id="all_branches_complete",
        trigger_rule="all_success"   # alerts only if everything succeeded
    )

    # -------------------------------
    # 5️⃣ Alerts
    # -------------------------------
    def run_alert_worker():
        from alerts.alert_worker import run
        run()

    alerts = PythonOperator(
        task_id="alerts",
        python_callable=run_alert_worker,
    )

    end = EmptyOperator(task_id="end")

    # -------------------------------
    # 🔗 Dependency Graph
    # -------------------------------

    start >> ingestion

    # Forecast chain
    ingestion >> demand_features >> demand_forecasting >> inventory_optimization

    # Parallel branch
    ingestion >> customer_risk
    ingestion >> multi_warehouse >> network_optimization

    # Join everything
    [
        inventory_optimization,
        customer_risk,
        network_optimization
    ] >> all_done

    all_done >> alerts >> end