# src/knnbot/utils/telegram.py
import requests
import logging
import os

logger = logging.getLogger(__name__)


def send_message(bot_token, chat_id, message):
    if not bot_token or not chat_id:
        logger.warning("Telegram Bot-Token oder Chat-ID nicht konfiguriert.")
        return
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    escaped = message
    for char in escape_chars:
        escaped = escaped.replace(char, f'\\{char}')
    api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': escaped, 'parse_mode': 'MarkdownV2'}
    try:
        response = requests.post(api_url, data=payload, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Fehler beim Senden der Telegram-Nachricht: {e}")
    except Exception as e:
        logger.error(f"Unerwarteter Fehler (Telegram): {e}")


def send_photo(bot_token, chat_id, file_path, caption=""):
    if not bot_token or not chat_id:
        return
    api_url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    try:
        with open(file_path, 'rb') as img:
            response = requests.post(api_url,
                                     data={'chat_id': chat_id, 'caption': caption},
                                     files={'photo': img}, timeout=30)
            response.raise_for_status()
    except FileNotFoundError:
        logger.error(f"Bild nicht gefunden: {file_path}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Fehler beim Senden des Fotos: {e}")
    except Exception as e:
        logger.error(f"Unerwarteter Fehler beim Foto-Versand: {e}")


def send_document(bot_token, chat_id, file_path, caption=""):
    if not bot_token or not chat_id:
        return
    if not os.path.exists(file_path):
        logger.error(f"Datei nicht gefunden: {file_path}")
        return
    api_url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    try:
        with open(file_path, 'rb') as f:
            requests.post(api_url, data={'chat_id': chat_id, 'caption': caption},
                          files={'document': (os.path.basename(file_path), f)}, timeout=30)
    except Exception as e:
        logger.error(f"Fehler beim Senden des Dokuments via Telegram: {e}")
