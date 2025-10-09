from airflow import DAG
from airflow.operators.python import PythonOperator, PythonVirtualenvOperator
from datetime import datetime, timedelta
from airflow.utils.task_group import TaskGroup

# Import project modules
from ingestion.main import run_ingestion
from analytics.pipeline import run_pipeline

# -------------------------------
# Failure alert callback
# -------------------------------
def failure_alert(context):
    task = context.get('task_instance').task_id
    dag_id = context.get('task_instance').dag_id
    execution_date = context.get('execution_date')
    print(f"[ALERT] Task {task} in DAG {dag_id} failed on {execution_date}!")

# -------------------------------
# Default DAG arguments
# -------------------------------
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': failure_alert,
}

# -------------------------------
# DAG Definition
# -------------------------------
with DAG(
    dag_id='supplychain_pipeline_dag',
    default_args=default_args,
    description='Supply Chain ETL, Analytics, and Alerts',
    schedule_interval='*/30 * * * *',
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['supply_chain', 'etl', 'analytics', 'alerts'],
) as dag:

    # -------------------------------
    # Task 1: Data Ingestion
    # -------------------------------
    ingestion_task = PythonOperator(
        task_id='ingestion',
        python_callable=run_ingestion,
        op_kwargs={'env': 'default', 'once': True},
        provide_context=True,
        doc_md="Fetches and loads data into the system.",
    )

    # -------------------------------
    # Task Group: Analytics + Alerts
    # -------------------------------
    with TaskGroup('analytics_alerts_group', tooltip="Analytics and Alerts") as analytics_alerts_group:

        analytics_task = PythonOperator(
            task_id='analytics',
            python_callable=run_pipeline,
            provide_context=True,
            doc_md="Runs ML forecasting and analytics.",
        )

        # Alerts
        def run_alert_worker():
            from alerts.alert_worker import run_alert_checks
            run_alert_checks()

        alerts_task = PythonVirtualenvOperator(
            task_id="alerts",
            python_callable=run_alert_worker,
            requirements=[
               "sqlalchemy>=2.0",
               "pandas>=1.2.5,<2.2",
               "numpy>=1.26.0",
               "python-dotenv>=1.0.0",
               "psycopg2-binary>=2.9.0",
               "plotly>=5.0.0"
            ],
            system_site_packages=False,
            python_version="3.12",
            doc_md="Performs alert checks and notifications (isolated venv).",
        )

        analytics_task >> alerts_task

    # -------------------------------
    # Set DAG execution order
    # -------------------------------
    ingestion_task >> analytics_alerts_group
