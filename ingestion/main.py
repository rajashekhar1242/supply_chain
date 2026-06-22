# ingestion/main.py
import time
import os

from ingestion.config import load_settings
from ingestion.mail_handler import (
    connect_mailbox,
    fetch_emails_by_subject,
    process_email,
)
from ingestion.file_processor import (
    save_attachment,
    read_file,
    normalize_dates,
    generate_file_hash,
    PROCESSED,
    ERROR,
    safe_move,
)
from ingestion.db_handler import init_db, insert_records, check_duplicate
from ingestion.utils import log


def process_file(part, eid, subject, sender, supabase):
    path = save_attachment(part, eid)
    if not path:
        return None

    fhash = generate_file_hash(path)
    fname = os.path.basename(path)

    # ✅ Duplicate Check
    if supabase and check_duplicate(supabase, fhash):
        log.info(f"Duplicate skipped {fname}")
        safe_move(path, PROCESSED)
        return None

    try:
        df = normalize_dates(read_file(path))
        fname_lower = fname.lower()

        # 🔥 TABLE DETECTION
        if "fact_order_line" in fname_lower:
            table = "fact_order_line"

        elif "fact_aggregate" in fname_lower:
            table = "fact_aggregate"

        elif "weekly_supplier" in fname_lower:
            table = "weekly_supplier_updates"

        elif "weekly_transfer" in fname_lower:
            table = "weekly_transfer_updates"

        elif "weekly_cost" in fname_lower:
            table = "weekly_cost_updates"

        else:
            log.warning(f"Unknown file type {fname}, skipping")
            safe_move(path, ERROR)
            return None

        # ✅ Supabase insert
        insert_records(
            supabase,
            table,
            df.to_dict(orient="records"),
            fname,
            fhash,
            subject,
            sender,
        )

        safe_move(path, PROCESSED)
        return table

    except Exception as e:
        log.error(f"Failed {fname}: {e}", exc_info=True)
        safe_move(path, ERROR)
        return None


# 👇 Main callable for Airflow
def run_ingestion(env: str = "dev", once: bool = True):
    log.info("Ingestion pipeline started")

    settings = load_settings(env)
    supabase = init_db(settings)

    if not supabase:
        log.error("Supabase client not initialized")
        return

    try:
        mail = connect_mailbox(settings)

        ids = fetch_emails_by_subject(mail, settings.subject_allow)
        log.info(f"Matched {len(ids)} emails with allowed subjects")

        processed_tables = set()

        for eid in ids:
            result = process_email(
                mail,
                eid,
                settings.subject_allow,
                settings.expected_sender,
                lambda part,
                eid=eid,
                subject=settings.subject_allow,
                sender=settings.expected_sender:
                process_file(part, eid, subject, sender, supabase),
            )

            if result:
                processed_tables.add(result)

        if processed_tables:
            log.info(f"Tables updated this run: {processed_tables}")

            # 🔥 Trigger Demand Engine if sales tables updated
            if (
                "fact_order_line" in processed_tables
                or "fact_aggregate" in processed_tables
            ):
                log.info("Triggering Demand Engine...")
                from analytics.pipeline import run_pipeline
                run_pipeline(supabase, processed_tables)

        mail.close()
        mail.logout()

    except Exception as e:
        log.error(f"Ingestion error: {e}", exc_info=True)

    if not once:
        log.info(f"Sleeping {settings.poll_seconds}s before next cycle...")
        time.sleep(settings.poll_seconds)


if __name__ == "__main__":
    run_ingestion(env="dev", once=False)