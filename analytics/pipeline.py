# analytics/main.py
from ingestion.config import load_settings
from supabase import create_client
from analytics.forecasting import forecast_demand
from ingestion.utils import log
from analytics.kpi_analysis import get_kpi_summary
from analytics.customer_risk import run_customer_risk_pipeline


def run_pipeline():
    """Main analytics pipeline (Airflow-ready callable)."""
    log.info("Starting analytics pipeline")

    settings = load_settings()
    supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)

    log.info("Running Customer KPI Risk Predictions")
    run_customer_risk_pipeline()

    log.info("Running KPI analysis")
    kpi = get_kpi_summary(supabase)
    log.info(f"KPI analysis complete ({len(kpi)} records)")

    log.info("Running demand forecasting")
    forecasts = forecast_demand(supabase)
    log.info(f"Forecasting complete ({len(forecasts)} products)")

    log.info("Generating policy recommendations")
    # TODO: add logic here
    log.info("Pipeline completed successfully")

    return {"kpi_count": len(kpi), "forecast_count": len(forecasts)}


if __name__ == "__main__":
    # Manual run
    run_pipeline()
