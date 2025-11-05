import re


def clean_text(text: str) -> str:

    """
    This function cleans a text: 
    lowercase, save letters and numbers, delete double space.
    """

    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text


