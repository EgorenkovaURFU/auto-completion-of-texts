import re
import unicodedata

URL_RE = re.compile(r"https?://\S+")
USER_RE = re.compile(r"@\w+")


def clean_text(text):

    """
    This function cleans a text: 
    lowercase, save letters and numbers, delete double space.
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = URL_RE.sub(" <url> ", text)
    text = USER_RE.sub(" <user> ", text)
    text = re.sub(r"[^a-z0-9'\s]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text


