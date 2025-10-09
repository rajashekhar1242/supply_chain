from ingestion.logger import logger as log
import email
from email.header import decode_header

def decode_mime_header(header_val: str) -> str:
    try:
        parts = decode_header(header_val)
        decoded = "".join(
            part.decode(enc or "utf-8") if isinstance(part, bytes) else part
            for part, enc in parts
        )
        log.debug(f"Decoded header: {decoded}")
        return decoded
    except Exception as e:
        log.error(f"Failed to decode header: {header_val}", exc_info=True)
        return header_val
