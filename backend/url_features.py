from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urlsplit

URL_MODEL_FEATURE_ORDER = (
    "url_length", "host_length", "is_ip", "tld", "tld_length", "subdomain_count",
    "is_https", "port_present", "path_length", "path_depth", "query_length",
    "query_parameter_count", "fragment_present", "letter_count", "letter_ratio",
    "digit_count", "digit_ratio", "special_count", "special_ratio", "dot_count",
    "hyphen_count", "underscore_count", "equals_count", "question_count",
    "ampersand_count", "at_sign_count", "percent_escape_count", "host_digit_count",
    "host_hyphen_count", "host_label_count", "longest_host_label", "punycode_present",
    "url_entropy", "host_entropy",
)


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    size = len(value)
    return -sum((count / size) * math.log2(count / size) for count in counts.values())


def extract_url_model_features(canonical_url: str) -> dict[str, str | int | float]:
    """Create the second model's URL-only features from an already validated canonical URL.

    This is a project-defined deterministic schema. It intentionally does not use
    dataset-derived similarity, reputation, webpage, or label-informed columns.
    """
    if not isinstance(canonical_url, str) or not canonical_url:
        raise ValueError("A canonical URL is required.")
    parts = urlsplit(canonical_url)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise ValueError("A canonical HTTP(S) URL is required.")
    host = parts.hostname.rstrip(".").encode("idna").decode("ascii").lower()
    url = canonical_url
    try:
        ipaddress.ip_address(host.strip("[]"))
        is_ip = 1
        labels: list[str] = []
        tld = ""
        subdomains = 0
    except ValueError:
        is_ip = 0
        labels = host.split(".")
        tld = labels[-1] if labels else ""
        subdomains = max(len(labels) - 2, 0)
    try:
        explicit_port = parts.port is not None
    except ValueError as exc:
        raise ValueError("URL port is invalid.") from exc
    length = len(url)
    letters = sum(ch.isascii() and ch.isalpha() for ch in url)
    digits = sum(ch.isascii() and ch.isdigit() for ch in url)
    special = length - letters - digits
    path_segments = [segment for segment in parts.path.split("/") if segment]
    query_params = parts.query.count("&") + 1 if parts.query else 0
    host_digits = sum(ch.isascii() and ch.isdigit() for ch in host)
    return {
        "url_length": length,
        "host_length": len(host),
        "is_ip": is_ip,
        "tld": tld,
        "tld_length": len(tld),
        "subdomain_count": subdomains,
        "is_https": int(parts.scheme.lower() == "https"),
        "port_present": int(explicit_port),
        "path_length": len(parts.path),
        "path_depth": len(path_segments),
        "query_length": len(parts.query),
        "query_parameter_count": query_params,
        "fragment_present": int(bool(parts.fragment)),
        "letter_count": letters,
        "letter_ratio": letters / length if length else 0.0,
        "digit_count": digits,
        "digit_ratio": digits / length if length else 0.0,
        "special_count": special,
        "special_ratio": special / length if length else 0.0,
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "underscore_count": url.count("_"),
        "equals_count": url.count("="),
        "question_count": url.count("?"),
        "ampersand_count": url.count("&"),
        "at_sign_count": url.count("@"),
        "percent_escape_count": len(re.findall(r"%[0-9a-fA-F]{2}", url)),
        "host_digit_count": host_digits,
        "host_hyphen_count": host.count("-"),
        "host_label_count": len(labels),
        "longest_host_label": max(map(len, labels), default=0),
        "punycode_present": int(any(label.startswith("xn--") for label in labels)),
        "url_entropy": _entropy(url),
        "host_entropy": _entropy(host),
    }
