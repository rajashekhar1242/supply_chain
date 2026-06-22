import os
from pathlib import Path
import yaml
from dotenv import load_dotenv
from ingestion.utils import log

load_dotenv()


class Settings:
    def __init__(self, env: str = "default"):

        # defaults
        self.poll_seconds = 300
        self.batch_size = 500
        self.once = False
        self.imap_server = "imap.gmail.com"

        # load config.yaml if exists
        cfg_file = Path(__file__).parent.parent / "config.yaml"
        values = {}

        if cfg_file.exists():
            with open(cfg_file) as f:
                raw = yaml.safe_load(f) or {}
            values = (raw.get("default", {}) | raw.get(env, {}))

        # ==============================
        # Supabase Configuration
        # ==============================

        self.supabase_url = os.getenv("SUPABASE_URL", values.get("supabase_url"))
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY",
                                      values.get("supabase_service_role_key"))

        # ==============================
        # Email Configuration
        # ==============================

        self.expected_sender = os.getenv("EXPECTED_SENDER", values.get("expected_sender"))
        self.email_user = os.getenv("EMAIL_USER", values.get("email_user"))
        self.email_pass = os.getenv("EMAIL_PASS", values.get("email_pass"))

        self.smtp_host = os.getenv("SMTP_HOST", values.get("smtp_host"))
        self.smtp_port = int(os.getenv("SMTP_PORT", values.get("smtp_port") or 0)) or None
        self.smtp_user = os.getenv("SMTP_USER", values.get("smtp_user"))
        self.smtp_pass = os.getenv("SMTP_PASS", values.get("smtp_pass"))

        self.alert_from = os.getenv("ALERT_FROM", values.get("alert_from"))
        self.alert_to = os.getenv("ALERT_TO", values.get("alert_to"))

        # ==============================
        # Other Flags
        # ==============================

        self.duplicate_window_minutes = int(
            os.getenv("DUPLICATE_WINDOW_MINUTES",
                      values.get("duplicate_window_minutes") or 0)
        ) or None

        val = str(os.getenv("ALERT_ON_STOCKOUT",
                            values.get("alert_on_stockout"))).lower()
        self.alert_on_stockout = val in ["true", "1", "yes"]

        # subject_allow could be a comma string or list
        subject_val = os.getenv("SUBJECT_ALLOW",
                                values.get("subject_allow"))

        if isinstance(subject_val, str):
            cleaned = subject_val.strip("[]")
            self.subject_allow = [
                s.strip().strip('"').strip("'")
                for s in cleaned.split(",")
            ]
        else:
            self.subject_allow = subject_val

        # ==============================
        # Validation
        # ==============================

        if not self.supabase_url or not self.supabase_key:
            log.warning("⚠️ Supabase credentials missing")

        if not self.email_user or not self.email_pass:
            log.warning("⚠️ Email credentials missing")

        log.info(f"Settings loaded (env={env})")


def load_settings(env: str = "default") -> Settings:
    return Settings(env=env)