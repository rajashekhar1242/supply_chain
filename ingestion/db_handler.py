# ingestion/db_handler.py

from supabase import create_client, Client
from ingestion.utils import log

def init_db(settings) -> Client | None:
    """Initialize Supabase client"""
    try:
        if not settings.supabase_url or not settings.supabase_service_role_key:
            log.warning("No Supabase credentials")
            return None
        log.info("Initializing Supabase client")
        return create_client(settings.supabase_url, settings.supabase_service_role_key)
    except Exception as e:
        log.error(f"Supabase init failed: {e}")
        return None


def check_duplicate(supabase, fhash: str) -> bool:
    """Check if file hash already exists in processed_files"""
    try:
        result = supabase.table("processed_files").select("id").eq("file_hash", fhash).execute()
        return len(result.data) > 0
    except Exception as e:
        log.error(f"Duplicate check failed: {e}")
        return False


def insert_records(supabase, table: str, records: list, file_name: str, fhash: str, subject: str, sender: str):
    """Insert into target fact table, then log the file in processed_files"""
    if not records:
        return

    try:
        supabase.table(table).insert(records).execute()

        supabase.table("processed_files").insert({
            "filename": file_name,
            "file_hash": fhash,
            "target_table": table,
            "email_subject": subject,
            "email_sender": sender
        }).execute()

        log.info(f"Inserted {len(records)} rows into {table} and logged {file_name} in processed_files")
    except Exception as e:
        log.error(f"Insert failed for {table}: {e}")
