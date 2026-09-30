"""
text_preprocessing.py - preprocessing for short, noisy, opinionated text
(tweets, support tickets, product reviews).

Philosophy: DON'T OVER-CLEAN. Every step below removes information, so each
is opt-in and tied to a specific downstream model.

    mode       what it does                                   use it for
    ---------  ---------------------------------------------  -----------------------------
    minimal    normalise URLs/@mentions/whitespace only       TF-IDF (it lowercases itself),
               (returns text, not tokens)                     transformers, anything with its
                                                              own tokenizer
    tokens     spaCy tokens, lowercased, punctuation dropped  Word2Vec / embeddings
               (keeps "!" and "?")
    lemma      tokens -> lemmas, stopwords KEPT               linear models on small data
    content    lemma + stopwords removed, negations KEPT      topic models, keyword analysis

Rules of thumb:
  * Tokenization  - always matters. Tweets need social-aware handling
                    (emoji, "n't", hashtags), so plain str.split() is not enough.
  * Lemmatization - helps when data is small (merges "delayed/delays/delay");
                    matters little for large data or transformers.
  * Stopwords     - dangerous for sentiment: "not", "no", "never", "too" carry
                    the signal. Removing default stopword lists turns
                    "not good" into "good". We protect them via KEEP_STOPWORDS.
  * POS tagging   - rarely a model input for classification; mainly useful for
                    analysis (which adjectives/verbs drive a class?) and for
                    targeted filtering (e.g. keep only nouns+adjectives).
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Iterable, Iterator, List, Tuple

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_RE = re.compile(r"\s+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
REPEAT_RE = re.compile(r"(.)\1{2,}")       # "soooo" -> "soo"
WS_RE = re.compile(r"\s+")

URL_TOKEN = "xxurl"
USER_TOKEN = "xxuser"

# Stopwords that must survive stopword removal (negation / intensity / contrast).
KEEP_STOPWORDS = {
    "no", "not", "nor", "never", "n't", "cannot", "none", "nothing", "nobody",
    "neither", "without", "against", "but", "however", "very", "too", "really",
    "so", "only", "more", "most", "less", "few", "again", "n’t",
}

# Punctuation that carries emotion and is worth keeping.
KEEP_PUNCT = {"!", "?"}

MODES = ("minimal", "tokens", "lemma", "content")


# --------------------------------------------------------------------------
# Step 0: social-media-aware normalisation (safe for every model)
# --------------------------------------------------------------------------
def clean_social(
    text: str,
    mask_mentions: bool = True,
    keep_hashtag_text: bool = True,
    squeeze_repeats: bool = True,
    unescape_html: bool = True,
) -> str:
    """Light normalisation. Does NOT lowercase, stem, or remove words."""
    if not isinstance(text, str):
        return ""
    if unescape_html:
        import html
        text = html.unescape(text)          # "&amp;" -> "&"
    text = URL_RE.sub(f" {URL_TOKEN} ", text)
    if mask_mentions:
        text = MENTION_RE.sub(f" {USER_TOKEN} ", text)
    if keep_hashtag_text:
        text = HASHTAG_RE.sub(r"\1", text)  # "#fail" -> "fail"
    if squeeze_repeats:
        text = REPEAT_RE.sub(r"\1\1", text) # keep a hint of emphasis
    return WS_RE.sub(" ", text).strip()


# --------------------------------------------------------------------------
# spaCy loader
# --------------------------------------------------------------------------
@lru_cache(maxsize=2)
def get_nlp(model: str = "en_core_web_sm"):
    """Load spaCy once. NER + parser disabled: we don't need them, ~3-5x faster."""
    import spacy

    try:
        return spacy.load(model, disable=["ner", "parser"])
    except OSError as e:
        raise OSError(
            f"spaCy model '{model}' not found. Run:\n"
            f"    python -m spacy download {model}"
        ) from e


# --------------------------------------------------------------------------
# Core: text -> tokens
# --------------------------------------------------------------------------
def _doc_to_tokens(doc, mode: str) -> List[str]:
    out: List[str] = []
    for t in doc:
        if t.is_space:
            continue
        if t.is_punct and t.text not in KEEP_PUNCT:
            continue
        if mode == "content" and t.is_stop and t.lower_ not in KEEP_STOPWORDS:
            continue
        out.append(t.lemma_.lower() if mode in ("lemma", "content") else t.lower_)
    return out


def tokenize_many(
    texts: Iterable[str],
    mode: str = "lemma",
    n_process: int = 1,
    batch_size: int = 512,
) -> Iterator[List[str]]:
    """Yield a token list per input text. Use this for big corpora (nlp.pipe)."""
    if mode not in MODES or mode == "minimal":
        raise ValueError(f"mode must be one of {MODES[1:]} for token output, got {mode!r}")
    nlp = get_nlp()
    cleaned = (clean_social(t) for t in texts)
    for doc in nlp.pipe(cleaned, batch_size=batch_size, n_process=n_process):
        yield _doc_to_tokens(doc, mode)


def preprocess(text: str, mode: str = "lemma") -> str:
    """Single text -> single string. Convenient, but slow in loops; use preprocess_many."""
    if mode == "minimal":
        return clean_social(text)
    return " ".join(next(tokenize_many([text], mode)))


def preprocess_many(texts: Iterable[str], mode: str = "lemma", **kw) -> List[str]:
    """Batch version of preprocess() - returns one string per input."""
    if mode == "minimal":
        return [clean_social(t) for t in texts]
    return [" ".join(toks) for toks in tokenize_many(texts, mode, **kw)]


# --------------------------------------------------------------------------
# Analysis helpers (POS, tokenizer comparison)
# --------------------------------------------------------------------------
def pos_tag(text: str) -> List[Tuple[str, str, str]]:
    """[(token, coarse POS, fine tag)] on lightly-cleaned text."""
    doc = get_nlp()(clean_social(text))
    return [(t.text, t.pos_, t.tag_) for t in doc if not t.is_space]


def pos_profile(texts: Iterable[str]) -> Counter:
    """Coarse-POS counts over a corpus (e.g. compare negative vs positive tweets)."""
    c: Counter = Counter()
    for doc in get_nlp().pipe((clean_social(t) for t in texts), batch_size=512):
        c.update(t.pos_ for t in doc if not t.is_space)
    return c


def compare_tokenizers(text: str) -> dict:
    """Same text through whitespace split, NLTK TweetTokenizer, and spaCy."""
    from nltk.tokenize import TweetTokenizer

    return {
        "whitespace": text.split(),
        "nltk_tweet": TweetTokenizer(preserve_case=False, reduce_len=True).tokenize(text),
        "spacy": [t.text for t in get_nlp()(text) if not t.is_space],
    }


if __name__ == "__main__":
    demo = "@united your flight was NOT good!!! delayed 3 hrs sooooo bad http://t.co/x #fail"
    for m in MODES:
        print(f"{m:8s} -> {preprocess(demo, m)}")
