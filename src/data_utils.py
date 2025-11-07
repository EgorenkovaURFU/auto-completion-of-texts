import re
import unicodedata


def clean_text(text):

    """
    This function cleans a text: 
    lowercase, save letters and numbers, delete double space.
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text


