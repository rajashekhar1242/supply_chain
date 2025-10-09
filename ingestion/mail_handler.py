# ingestion/mail_handler.py
import imaplib, ssl, re, email
from tenacity import retry, stop_after_attempt, wait_exponential
from ingestion.utils import log, decode_mime_header

@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=30))
def connect_mailbox(settings):
    log.info(f"Connecting to {settings.imap_server}...")
    ssl_context = ssl.create_default_context()
    mail = imaplib.IMAP4_SSL(settings.imap_server, ssl_context=ssl_context)
    mail.login(settings.email_user, settings.email_pass)
    mail.select("inbox")
    log.debug("Mailbox connection established")
    return mail

def fetch_emails_by_subject(mail, filters):
    log.info("Fetching emails started...")
    status, messages = mail.search(None, "UNSEEN")
    if status != "OK":
        log.warning("Failed to search inbox for UNSEEN messages")
        return []
    ids = messages[0].split()
    log.debug(f"Found {len(ids)} unseen emails")
    if not filters:
        return ids

    matched = []
    for eid in ids:
        status, data = mail.fetch(eid, "(BODY[HEADER.FIELDS (SUBJECT)])")
        if status != "OK" or not data or not data[0]:
            continue
        subj_line = data[0][1].decode("utf-8", errors="ignore")
        match = re.search(r"Subject: (.*)", subj_line, re.IGNORECASE)
        subject = match.group(1).strip() if match else subj_line
        subject_clean = re.sub(r"^(Re|Fw|Fwd):\s*", "", subject, flags=re.IGNORECASE)
        log.debug(f"Email {eid} subject: {subject_clean}")
        if any(f.lower() in subject_clean.lower() for f in filters):
            matched.append(eid)
    log.info(f"Matched {len(matched)} emails with filters")
    return matched

def process_email(mail, eid, filters, expected_sender, process_file_fn):
    status, data = mail.fetch(eid, "(RFC822)")
    if status != "OK":
        log.error(f"Failed to fetch email {eid}")
        return False
    msg = email.message_from_bytes(data[0][1])
    sender = decode_mime_header(msg.get("From", ""))
    subject = decode_mime_header(msg.get("Subject", ""))

    if expected_sender and expected_sender.lower() not in sender.lower():
        log.warning(f"Skipping {sender} for email {eid}")
        return False
    if filters and not any(f.lower() in subject.lower() for f in filters):
        log.debug(f"Skipping email {eid}, subject {subject} not matching filters")
        return False

    processed = []
    for part in msg.walk():
        if part.get_content_maintype() == "multipart":
            continue
        if part.get("Content-Disposition") is None:
            continue
        path = process_file_fn(part, eid, subject, sender)
        if path:
            processed.append(path)
    log.info(f"Email {eid} processed, {len(processed)} attachments handled")
    return bool(processed)
