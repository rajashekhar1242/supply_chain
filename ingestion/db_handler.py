# ingestion/db_handler.py

from supabase import create_client, Client
from dotenv import load_dotenv
import os
from ingestion.utils import log

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials missing in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def init_db(settings=None):
    """
    Supabase client is initialized globally.
    Keeping this function for compatibility.
    """
    log.info("Supabase client initialized")
    return supabase


def check_duplicate(client: Client, fhash: str) -> bool:
    """Check if file hash exists in processed_files"""
    try:
        response = (
            client
            .table("processed_files")
            .select("file_hash")
            .eq("file_hash", fhash)
            .limit(1)
            .execute()
        )

        return len(response.data) > 0

    except Exception as e:
        log.error(f"Duplicate check failed: {e}")
        return False


def insert_records(
    client: Client,
    table: str,
    records: list,
    file_name: str,
    fhash: str,
    subject: str,
    sender: str,
):
    """Insert records and log processed file"""

    if not records:
        return

    try:
        batch_size = 500

        # Batch insert
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]

            response = client.table(table).insert(batch).execute()

            if response.data is None:
                raise Exception(response)

        # Log processed file
        client.table("processed_files").insert({
            "filename": file_name,
            "file_hash": fhash,
            "target_table": table,
            "email_subject": subject,
            "email_sender": sender
        }).execute()

        log.info(f"Inserted {len(records)} rows into {table} and logged {file_name}")

    except Exception as e:
        log.error(f"Insert failed for {table}: {e}")