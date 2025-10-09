import pandas as pd
from ingestion.utils import log

def get_kpi_summary(supabase):
    """Fetch pre-aggregated KPI summary from DB (no recompute)."""
    try:
        result = supabase.table("kpi_summary").select("*").execute()
        df = pd.DataFrame(result.data)
    except Exception as e:
        log.error(f"Failed to fetch kpi_summary: {e}")
        return pd.DataFrame()

    if df.empty:
        log.warning("⚠️ No KPI summary data available")
    else:
        log.info(f"Fetched {len(df)} KPI summary rows")

    return df
 