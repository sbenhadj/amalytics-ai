"""
Format-Preserving Encryption (FPE) for medical document anonymization.

Aligned with the original FPE_encryption.ipynb notebook.
Supports:
- PDF text extraction
- NER-based name detection (CamemBERT)
- Regex-based pattern scrubbing (codes, dates, emails, phones, postal codes)
- Deterministic pseudonymization with pyffx
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyffx

# Optional imports for NER - will be loaded on demand
_ner_pipeline = None
_nltk_initialized = False


@dataclass
class AnonymizationConfig:
    """
    Configuration for document anonymization.
    
    Attributes:
        secret_key: Bytes used for deterministic format-preserving encryption.
        use_ner: Whether to use NER model for detecting person names.
        ner_model_path: Path to CamemBERT NER model (if use_ner=True).
        anonymize_codes: Anonymize long numeric codes.
        anonymize_dates: Anonymize dates (dd/mm/yyyy format).
        anonymize_emails: Anonymize email addresses.
        anonymize_phones: Anonymize French phone numbers.
        anonymize_postal_codes: Anonymize postal codes.
    """
    secret_key: bytes = b"sbh86"
    use_ner: bool = True
    ner_model_path: str = ""
    anonymize_codes: bool = True
    anonymize_dates: bool = True
    anonymize_emails: bool = True
    anonymize_phones: bool = True
    anonymize_postal_codes: bool = True


@dataclass
class _EncryptorCache:
    """Cache for FPE encryptors and memoized values."""
    numeric: dict[int, pyffx.Integer] = field(default_factory=dict)
    alpha: dict[str, pyffx.String] = field(default_factory=dict)
    memo: dict[str, str] = field(default_factory=dict)


def _get_numeric_encryptor(cache: _EncryptorCache, secret_key: bytes, length: int) -> pyffx.Integer:
    """Get or create a numeric FPE encryptor for the given length."""
    if length not in cache.numeric:
        cache.numeric[length] = pyffx.Integer(secret_key, length=length)
    return cache.numeric[length]


def _get_alpha_encryptor(cache: _EncryptorCache, secret_key: bytes, alphabet: str, length: int) -> pyffx.String:
    """Get or create an alphabetic FPE encryptor for the given alphabet and length."""
    key = f"{alphabet}_{length}"
    if key not in cache.alpha:
        cache.alpha[key] = pyffx.String(secret_key, alphabet=alphabet, length=length)
    return cache.alpha[key]


def fpe_encrypt(value: str, secret_key: bytes, cache: _EncryptorCache) -> str:
    """
    Format-preserving encryption of a value.
    
    Uses dynamic alphabet extraction like the original notebook.
    
    Args:
        value: The string to encrypt.
        secret_key: Secret key for FPE.
        cache: Encryptor cache for memoization.
    
    Returns:
        Encrypted string with same format as input.
    """
    if value in cache.memo:
        return cache.memo[value]
    
    try:
        if value.isdigit():
            # Numeric value
            enc = _get_numeric_encryptor(cache, secret_key, len(value))
            encrypted = str(enc.encrypt(int(value))).zfill(len(value))
        
        elif re.match(r'^\d{2}/\d{2}/\d{2,4}$', value):
            # Date format dd/mm/yyyy - encrypt each part separately
            parts = value.split('/')
            encrypted_parts = [
                str(_get_numeric_encryptor(cache, secret_key, len(p)).encrypt(int(p))).zfill(len(p))
                for p in parts
            ]
            encrypted = '/'.join(encrypted_parts)
        
        else:
            # Dynamic alphabet extraction (like original)
            alphabet = ''.join(sorted(set(value)))
            if len(alphabet) < 2:
                alphabet = alphabet + " "  # pyffx needs at least 2 chars
            enc = _get_alpha_encryptor(cache, secret_key, alphabet, len(value))
            encrypted = enc.encrypt(value)
    
    except Exception as e:
        print(f"[!] FPE error on '{value}': {e}")
        encrypted = value
    
    cache.memo[value] = encrypted
    return encrypted


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extract plain text from a PDF using pdfplumber.
    
    Args:
        pdf_path: Path to the PDF file.
    
    Returns:
        Extracted text content.
    """
    import pdfplumber
    
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def _init_nltk():
    """Initialize NLTK resources for sentence tokenization."""
    global _nltk_initialized
    if not _nltk_initialized:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        _nltk_initialized = True


def _get_ner_pipeline(model_path: str = ""):
    """
    Get or create the NER pipeline for person name detection.
    
    Uses CamemBERT-based NER model.
    """
    global _ner_pipeline
    
    if _ner_pipeline is not None:
        return _ner_pipeline
    
    try:
        from transformers import pipeline, CamembertTokenizer, CamembertForTokenClassification
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_path and os.path.exists(model_path):
            model = CamembertForTokenClassification.from_pretrained(model_path)
            tokenizer = CamembertTokenizer.from_pretrained(model_path)
        else:
            # Use default camembert-ner from HuggingFace
            model = "Jean-Baptiste/camembert-ner"
            tokenizer = model
            model = CamembertForTokenClassification.from_pretrained(model)
            tokenizer = CamembertTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
        
        _ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=device,
        )
        return _ner_pipeline
    
    except ImportError:
        print("[WARNING] transformers not installed. NER disabled.")
        return None
    except Exception as e:
        print(f"[WARNING] Could not load NER model: {e}")
        return None


def _normalize_whitespace(s: str) -> str:
    """Collapse all whitespace into single spaces."""
    return re.sub(r'\s+', ' ', s).strip()


def encrypt_ner_entities(text: str, secret_key: bytes, cache: _EncryptorCache, ner_model_path: str = "") -> str:
    """
    Detect and encrypt person names using NER.
    
    Args:
        text: Input text to process.
        secret_key: Secret key for FPE.
        cache: Encryptor cache.
        ner_model_path: Path to NER model.
    
    Returns:
        Text with person names encrypted.
    """
    _init_nltk()
    from nltk.tokenize import sent_tokenize
    
    ner = _get_ner_pipeline(ner_model_path)
    if ner is None:
        return text
    
    sentences = sent_tokenize(text, language="french")
    encrypted_map: dict[str, str] = {}
    
    for sentence in sentences:
        try:
            entities = ner(sentence)
            for entity in entities:
                if entity["entity_group"] in ["PER"]:
                    original = _normalize_whitespace(entity["word"])
                    if original and original not in encrypted_map:
                        encrypted_map[original] = fpe_encrypt(original, secret_key, cache)
        except Exception as e:
            print(f"[WARNING] NER error on sentence: {e}")
            continue
    
    # Replace all occurrences with encrypted versions
    for original, encrypted in encrypted_map.items():
        # Pattern handles multi-word names with variable whitespace
        pattern = r'\b' + r'\s*'.join(map(re.escape, original.split())) + r'\b'
        text = re.sub(pattern, encrypted, text, flags=re.IGNORECASE)
    
    return text


def anonymize_text(text: str, config: AnonymizationConfig | None = None) -> str:
    """
    Anonymize text content using FPE without file I/O.
    
    This is a convenience function for anonymizing text strings directly,
    useful when you already have the text and don't need file operations.
    
    Args:
        text: Input text to anonymize.
        config: Anonymization configuration.
    
    Returns:
        Anonymized text content.
    """
    cfg = config or AnonymizationConfig()
    cache = _EncryptorCache()
    
    # NER-based name encryption
    if cfg.use_ner:
        text = encrypt_ner_entities(text, cfg.secret_key, cache, cfg.ner_model_path)
    
    # Codes (8+ digits or mixed alphanumeric codes)
    if cfg.anonymize_codes:
        text = re.sub(
            r'\b(?:\d{8,}|\d{6,}[A-Z]\d{3,})\b',
            lambda m: fpe_encrypt(m.group(), cfg.secret_key, cache),
            text
        )
    
    # Dates (birth dates with various patterns)
    if cfg.anonymize_dates:
        date_patterns = [
            r'(?i)\b(date de naissance[:\s]*)\d{2}/\d{2}/\d{2,4}',
            r'(?i)\b(né\(e\)? le[:\s]*)\d{2}/\d{2}/\d{2,4}',
            r'(?i)\b(date de naissance[:\s]*)\d{1,2}\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}',
            r'(?i)\b(né\(e\)? le[:\s]*)\d{1,2}\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}',
        ]
        for pattern in date_patterns:
            text = re.sub(
                pattern,
                lambda m: m.group(1) + fpe_encrypt(m.group(0).replace(m.group(1), ''), cfg.secret_key, cache),
                text
            )
    
    # Email addresses
    if cfg.anonymize_emails:
        text = re.sub(
            r'\b[\w\.-]+@[\w\.-]+\.\w+\b',
            lambda m: fpe_encrypt(m.group(), cfg.secret_key, cache),
            text
        )
    
    # French phone numbers
    if cfg.anonymize_phones:
        text = re.sub(
            r'\b(?:0\d(?:[ .-]?\d{2}){4})\b',
            lambda m: fpe_encrypt(m.group(), cfg.secret_key, cache),
            text
        )
    
    # Postal codes with city names
    if cfg.anonymize_postal_codes:
        text = re.sub(
            r'\b\d{5} [A-Z]+\b',
            lambda m: fpe_encrypt(m.group(), cfg.secret_key, cache),
            text
        )
    
    return text


def encrypt_document(
    pdf_path: str | Path,
    output_folder: str | Path,
    config: AnonymizationConfig | None = None,
) -> str:
    """
    Anonymize a PDF document using FPE.
    
    This is the main function aligned with the original encrypt_doc() from the notebook.
    
    Args:
        pdf_path: Path to input PDF file.
        output_folder: Directory to save the encrypted text file.
        config: Anonymization configuration.
    
    Returns:
        The encrypted text content.
    """
    cfg = config or AnonymizationConfig()
    cache = _EncryptorCache()
    
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # NER-based name encryption
    if cfg.use_ner:
        text = encrypt_ner_entities(text, cfg.secret_key, cache, cfg.ner_model_path)
    
    # Codes (8+ digits or mixed alphanumeric codes)
    if cfg.anonymize_codes:
        text = re.sub(
            r'\b(?:\d{8,}|\d{6,}[A-Z]\d{3,})\b',
            lambda m: fpe_encrypt(m.group(), cfg.secret_key, cache),
            text
        )
    
    # Dates (birth dates with various patterns)
    if cfg.anonymize_dates:
        date_patterns = [
            r'(?i)\b(date de naissance[:\s]*)\d{2}/\d{2}/\d{2,4}',
            r'(?i)\b(né\(e\)? le[:\s]*)\d{2}/\d{2}/\d{2,4}',
            r'(?i)\b(date de naissance[:\s]*)\d{1,2}\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}',
            r'(?i)\b(né\(e\)? le[:\s]*)\d{1,2}\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}',
        ]
        for pattern in date_patterns:
            text = re.sub(
                pattern,
                lambda m: m.group(1) + fpe_encrypt(m.group(0).replace(m.group(1), ''), cfg.secret_key, cache),
                text
            )
    
    # Email addresses
    if cfg.anonymize_emails:
        text = re.sub(
            r'\b[\w\.-]+@[\w\.-]+\.\w+\b',
            lambda m: fpe_encrypt(m.group(), cfg.secret_key, cache),
            text
        )
    
    # French phone numbers
    if cfg.anonymize_phones:
        text = re.sub(
            r'\b(?:0\d(?:[ .-]?\d{2}){4})\b',
            lambda m: fpe_encrypt(m.group(), cfg.secret_key, cache),
            text
        )
    
    # Postal codes with city names
    if cfg.anonymize_postal_codes:
        text = re.sub(
            r'\b\d{5} [A-Z]+\b',
            lambda m: fpe_encrypt(m.group(), cfg.secret_key, cache),
            text
        )
    
    # Save to output file
    pdf_path = Path(pdf_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    base_name = pdf_path.stem
    output_filename = f"{base_name}_encrypted.txt"
    output_path = output_folder / output_filename
    
    with output_path.open("w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"✅ {output_path} created")
    return text


def anonymize_record(record: dict[str, Any], config: AnonymizationConfig | None = None) -> dict[str, Any]:
    """
    Apply FPE to selected string fields of a structured record.
    
    This function is for anonymizing JSON/dict records, not PDF documents.
    Use encrypt_document() for PDF anonymization.
    
    Args:
        record: Input dictionary.
        config: Anonymization configuration.
    
    Returns:
        A new dictionary with string values anonymized.
    """
    cfg = config or AnonymizationConfig()
    cache = _EncryptorCache()
    
    def anonymize_value(value: Any) -> Any:
        if isinstance(value, str) and value:
            return fpe_encrypt(value, cfg.secret_key, cache)
        elif isinstance(value, dict):
            return {k: anonymize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [anonymize_value(item) for item in value]
        else:
            return value
    
    return anonymize_value(record)


# Convenience function for batch processing
def encrypt_documents_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    config: AnonymizationConfig | None = None,
) -> list[str]:
    """
    Process all PDF files in a directory.
    
    Args:
        input_dir: Directory containing PDF files.
        output_dir: Directory to save encrypted text files.
        config: Anonymization configuration.
    
    Returns:
        List of output file paths.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_files = []
    
    for pdf_file in input_dir.glob("*.pdf"):
        print(f"Processing: {pdf_file}")
        encrypt_document(pdf_file, output_dir, config)
        output_files.append(str(output_dir / f"{pdf_file.stem}_encrypted.txt"))
    
    return output_files
