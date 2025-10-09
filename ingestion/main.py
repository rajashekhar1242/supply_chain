# ingestion/main.py
import time, os
from ingestion.config import load_settings
from ingestion.mail_handler import connect_mailbox, fetch_emails_by_subject, process_email
from ingestion.file_processor import save_attachment, read_file, normalize_dates, generate_file_hash, PROCESSED, ERROR, safe_move
from ingestion.db_handler import init_db, insert_records, check_duplicate
from ingestion.utils import log


def process_file(part, eid, subject, sender, supabase):
    path = save_attachment(part, eid)
    fhash = generate_file_hash(path)
    fname = os.path.basename(path)

    if supabase and check_duplicate(supabase, fhash):
        log.info(f"Duplicate skipped {fname}")
        safe_move(path, PROCESSED)
        return None

    try:
        df = normalize_dates(read_file(path))

        if "fact_order_line" in fname.lower():
            table = "fact_order_line"
        elif "fact_aggregate" in fname.lower():
            table = "fact_aggregate"
        else:
            log.warning(f"Unknown file type {fname}, skipping")
            return None

        insert_records(
            supabase,
            table,
            df.to_dict(orient="records"),
            fname,
            fhash,
            subject,
            sender
        )
        safe_move(path, PROCESSED)
        return path

    except Exception as e:
        log.error(f"Failed {fname}: {e}")
        os.rename(path, os.path.join(ERROR, fname))
        return None


# 👇 main callable for Airflow
def run_ingestion(env: str = "dev", once: bool = True):
    """Run ingestion once (used by Airflow)"""
    log.info("Ingestion pipeline started")
    settings = load_settings(env)
    supabase = init_db(settings)

    try:
        mail = connect_mailbox(settings)
        ids = fetch_emails_by_subject(mail, settings.subject_allow)
        log.info(f"Matched {len(ids)} emails with allowed subjects")
        processed_count = 0
        for eid in ids:
            if process_email(mail, eid, settings.subject_allow, settings.expected_sender,
                             lambda part, eid=eid, subject=settings.subject_allow, sender=settings.expected_sender:
                             process_file(part, eid, subject, sender, supabase)):
                processed_count += 1
        log.info(f"Processed {processed_count} attachments this run")
        mail.close()
        mail.logout()
    except Exception as e:
        log.error(f"Ingestion error: {e}", exc_info=True)

    if not once:
        # If you want old behavior (loop forever) run outside Airflow
        log.info(f"Sleeping {settings.poll_seconds}s before next cycle...")
        time.sleep(settings.poll_seconds)


if __name__ == "__main__":
    # for local/manual testing
    run_ingestion(env="dev", once=False)
