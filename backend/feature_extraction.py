from __future__ import annotations

import ipaddress
import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
import urllib3
from requests.adapters import HTTPAdapter

MAX_URL_LENGTH = 2048
MAX_HTML_BYTES = 1_500_000
MAX_REDIRECTS = 4
TOTAL_FETCH_SECONDS = 14.0
DNS_TIMEOUT_SECONDS = 3.0
_DNS_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="phishguard-dns")
CONNECT_TIMEOUT = 3.05
READ_TIMEOUT = 5.0
METADATA_HOSTS = {
    "metadata.google.internal", "metadata.google", "instance-data.ec2.internal",
    "metadata.azure.internal", "metadata.tencentyun.com", "metadata.oraclecloud.com",
}
BLOCKED_SUFFIXES = (".localhost", ".local", ".internal", ".test", ".invalid")

FEATURE_ORDER: list[str] = []  # Loaded and verified from feature_order.pkl at model startup.


class UnsafeTarget(ValueError):
    """A URL violates the scanner's destination or scheme policy."""


class DNSResolutionError(RuntimeError):
    """A public hostname could not be resolved."""


class FetchTimeoutError(RuntimeError):
    """DNS, connection, read, or total scan time limit was exceeded."""


class WebsiteFetchError(RuntimeError):
    """A remote response could not be retrieved or was unusable."""


@dataclass
class PageFetch:
    final_url: str
    html: str
    redirects: int
    self_redirects: int
    content_type: str
    warnings: list[str]
    status_code: int | None = None


def _normalize_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UnsafeTarget("A URL is required.")
    value = value.strip()
    if len(value) > MAX_URL_LENGTH:
        raise UnsafeTarget(f"URL must be {MAX_URL_LENGTH} characters or fewer.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise UnsafeTarget("URL contains invalid control characters.")
    try:
        parts = urlsplit(value)
    except ValueError:
        raise UnsafeTarget("URL syntax is invalid.") from None
    if parts.scheme.lower() not in {"http", "https"}:
        raise UnsafeTarget("Only http:// and https:// URLs are supported.")
    if not parts.hostname or parts.username is not None or parts.password is not None:
        raise UnsafeTarget("URL must have a hostname and cannot contain credentials.")
    try:
        host = parts.hostname.rstrip(".").encode("idna").decode("ascii").lower()
        port = parts.port
    except (UnicodeError, ValueError):
        raise UnsafeTarget("URL hostname or port is invalid.") from None
    if not host:
        raise UnsafeTarget("URL hostname is missing.")
    if port is not None and port != (443 if parts.scheme.lower() == "https" else 80):
        raise UnsafeTarget("Only the standard HTTP and HTTPS ports are allowed.")
    try:
        address = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        address = None
    if address is not None:
        _require_public_ip(address)
    else:
        if host in METADATA_HOSTS or any(host.endswith(s) for s in BLOCKED_SUFFIXES):
            raise UnsafeTarget("Local, internal, and metadata hosts are blocked.")
        if host == "localhost" or "." not in host:
            raise UnsafeTarget("A public fully qualified hostname is required.")
        if not re.fullmatch(r"(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", host):
            raise UnsafeTarget("URL hostname is invalid.")
    host_part = f"[{host}]" if ":" in host else host
    netloc = host_part if port is None else f"{host_part}:{port}"
    return urlunsplit((parts.scheme.lower(), netloc, parts.path or "/", parts.query, parts.fragment))


def _require_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    if not ip.is_global or ip.is_multicast or ip.is_unspecified or ip.is_reserved:
        raise UnsafeTarget("Private, loopback, link-local, and reserved IP addresses are blocked.")


def _resolve_public(host: str, port: int, timeout: float = DNS_TIMEOUT_SECONDS) -> str:
    try:
        direct = ipaddress.ip_address(host)
    except ValueError:
        direct = None
    if direct is not None:
        _require_public_ip(direct)
        return direct.compressed
    if host in METADATA_HOSTS or any(host.endswith(s) for s in BLOCKED_SUFFIXES):
        raise UnsafeTarget("Local, internal, and metadata hosts are blocked.")
    future = _DNS_EXECUTOR.submit(socket.getaddrinfo, host, port, 0, socket.SOCK_STREAM)
    try:
        answers = future.result(timeout=timeout)
    except FutureTimeoutError:
        future.cancel()
        raise FetchTimeoutError("DNS lookup timed out.") from None
    except (OSError, socket.gaierror):
        raise DNSResolutionError("The public hostname could not be resolved.") from None
    ips = list(dict.fromkeys(item[4][0].split("%", 1)[0] for item in answers))
    if not ips:
        raise UnsafeTarget("The hostname did not resolve to an address.")
    parsed = []
    for item in ips:
        try:
            ip = ipaddress.ip_address(item)
        except ValueError:
            raise UnsafeTarget("Hostname resolved to an invalid address.") from None
        _require_public_ip(ip)
        parsed.append(ip)
    # Pin the connection to a prevalidated address to prevent a second DNS lookup/rebinding.
    return parsed[0].compressed


class _PinnedAdapter(HTTPAdapter):
    def __init__(self, hostname: str, ip: str, scheme: str, port: int):
        super().__init__(max_retries=0, pool_connections=1, pool_maxsize=1)
        self.hostname, self.ip, self.scheme, self.port = hostname, ip, scheme, port

    def get_connection(self, url, proxies=None):
        if proxies:
            raise UnsafeTarget("Proxy use is disabled for URL scanning.")
        parsed = urlsplit(url)
        if parsed.hostname != self.hostname or parsed.scheme != self.scheme:
            raise UnsafeTarget("Connection target changed after validation.")
        if self.scheme == "https":
            return urllib3.HTTPSConnectionPool(
                host=self.ip, port=self.port, server_hostname=self.hostname,
                assert_hostname=self.hostname, cert_reqs="CERT_REQUIRED", maxsize=1,
            )
        return urllib3.HTTPConnectionPool(host=self.ip, port=self.port, maxsize=1)

    def send(self, request, **kwargs):
        default_port = 443 if self.scheme == "https" else 80
        host_header = f"[{self.hostname}]" if ":" in self.hostname else self.hostname
        request.headers["Host"] = host_header if self.port == default_port else f"{host_header}:{self.port}"
        return super().send(request, **kwargs)


def _request_once(url: str, deadline: float) -> requests.Response:
    parts = urlsplit(url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise FetchTimeoutError("Website analysis exceeded the total time limit.")
    ip = _resolve_public(parts.hostname or "", port, timeout=min(DNS_TIMEOUT_SECONDS, remaining))
    session = requests.Session()
    session.trust_env = False
    adapter = _PinnedAdapter(parts.hostname or "", ip, parts.scheme, port)
    session.mount(f"{parts.scheme}://", adapter)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        session.close()
        raise FetchTimeoutError("Website analysis exceeded the total time limit.")
    try:
        response = session.get(
            url, allow_redirects=False, stream=True,
            timeout=(min(CONNECT_TIMEOUT, remaining), min(READ_TIMEOUT, remaining)),
            headers={"User-Agent": "PhishGuard-Research-Scanner/1.0", "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1"},
        )
        response._phishguard_session = session
        return response
    except Exception:
        session.close()
        raise


def fetch_page(input_url: str) -> PageFetch:
    current = _normalize_url(input_url)
    deadline = time.monotonic() + TOTAL_FETCH_SECONDS
    redirects = 0
    self_redirects = 0
    warnings: list[str] = []
    for hop in range(MAX_REDIRECTS + 1):
        try:
            response = _request_once(current, deadline)
        except (UnsafeTarget, DNSResolutionError, FetchTimeoutError):
            raise
        except (requests.Timeout, urllib3.exceptions.TimeoutError, TimeoutError) as exc:
            raise FetchTimeoutError(f"Website request timed out ({type(exc).__name__}).") from None
        except (requests.RequestException, urllib3.exceptions.HTTPError, OSError) as exc:
            raise WebsiteFetchError(f"Website could not be fetched safely ({type(exc).__name__}).") from None
        try:
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                if not location or hop >= MAX_REDIRECTS:
                    raise UnsafeTarget("Invalid or excessive redirect chain is blocked.")
                target = _normalize_url(urljoin(current, location))
                if urlsplit(current).scheme == "https" and urlsplit(target).scheme != "https":
                    raise UnsafeTarget("HTTPS-to-HTTP redirect downgrades are blocked.")
                # Host/IP/port policy is checked here; the next hop resolves and pins its own public address.
                if urlsplit(current).hostname == urlsplit(target).hostname:
                    self_redirects += 1
                current = target
                redirects += 1
                continue
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if response.status_code in {401, 403}:
                warnings.append(f"The website returned HTTP {response.status_code} (access denied). No webpage content was analyzed. Missing webpage-dependent features were not fabricated or imputed; if available, the separate URL-only model uses only the submitted URL.")
                return PageFetch(current, "", redirects, self_redirects, content_type, warnings, response.status_code)
            if not 200 <= response.status_code < 300:
                warnings.append(f"The website returned HTTP {response.status_code}; page content is unavailable. URL-only analysis may still be possible.")
                return PageFetch(current, "", redirects, self_redirects, content_type, warnings, response.status_code)
            if content_type not in {"text/html", "application/xhtml+xml"}:
                warnings.append("The final URL did not return HTML; webpage-derived signals are unavailable.")
                return PageFetch(current, "", redirects, self_redirects, content_type, warnings, response.status_code)
            length = response.headers.get("Content-Length")
            if length:
                try:
                    declared_length = int(length)
                except ValueError:
                    declared_length = None
            else:
                declared_length = None
            if declared_length is not None and declared_length > MAX_HTML_BYTES:
                warnings.append("HTML response exceeded the download limit; webpage-derived signals are unavailable.")
                return PageFetch(current, "", redirects, self_redirects, content_type, warnings, response.status_code)
            chunks: list[bytes] = []
            total = 0
            try:
                for chunk in response.iter_content(16_384):
                    if time.monotonic() > deadline:
                        raise FetchTimeoutError("Website analysis exceeded the total time limit.")
                    total += len(chunk)
                    if total > MAX_HTML_BYTES:
                        warnings.append("HTML response exceeded the download limit; webpage-derived signals are unavailable.")
                        return PageFetch(current, "", redirects, self_redirects, content_type, warnings, response.status_code)
                    chunks.append(chunk)
            except (requests.Timeout, urllib3.exceptions.TimeoutError, TimeoutError) as exc:
                raise FetchTimeoutError(f"Website response timed out ({type(exc).__name__}).") from None
            except (requests.RequestException, urllib3.exceptions.HTTPError, OSError) as exc:
                raise WebsiteFetchError(f"Website response could not be read ({type(exc).__name__}).") from None
            encoding = response.encoding or "utf-8"
            try:
                html = b"".join(chunks).decode(encoding, errors="replace")
            except LookupError:
                warnings.append("The response declared an unsupported text encoding; UTF-8 replacement decoding was used.")
                html = b"".join(chunks).decode("utf-8", errors="replace")
            return PageFetch(current, html, redirects, self_redirects, content_type, warnings, response.status_code)
        finally:
            response.close()
            getattr(response, "_phishguard_session", requests).close() if hasattr(response, "_phishguard_session") else None
    raise WebsiteFetchError("Website redirect limit exceeded.")


class _PageSignals(HTMLParser):
    def __init__(self, final_url: str):
        super().__init__(convert_charrefs=True)
        self.final_url = final_url
        self.title_chunks: list[str] = []
        self.in_title = False
        self.has_title = False
        self.has_description = False
        self.responsive: bool | None = None
        self.favicon: bool | None = None
        self.images = 0
        self.css = 0
        self.js = 0
        self.iframes = 0
        self.has_submit = False
        self.has_hidden = False
        self.has_password = False
        self.external_form = False
        self.self_refs = 0
        self.empty_refs = 0
        self.external_refs = 0

    def handle_starttag(self, tag: str, attrs):
        a = {k.lower(): v for k, v in attrs if k}
        tag = tag.lower()
        if tag == "title": self.in_title = True; self.has_title = True
        if tag == "meta":
            if (a.get("name") or "").lower() == "description" and a.get("content") is not None: self.has_description = True
            if (a.get("name") or "").lower() == "viewport": self.responsive = True
        if tag == "link":
            rel = (a.get("rel") or "").lower().split()
            if "stylesheet" in rel: self.css += 1
            if any(x in rel for x in ("icon", "shortcut")): self.favicon = True
        if tag == "img": self.images += 1
        if tag == "script" and a.get("src") is not None: self.js += 1
        if tag == "iframe": self.iframes += 1
        if tag == "button" or (tag == "input" and (a.get("type") or "").lower() in {"submit", "image"}): self.has_submit = True
        if tag == "input":
            kind = (a.get("type") or "text").lower()
            if kind == "hidden": self.has_hidden = True
            if kind == "password": self.has_password = True
        if tag == "form" and a.get("action"):
            try:
                action = urlsplit(urljoin(self.final_url, a["action"]))
            except ValueError:
                action = None
            if action and action.hostname and action.hostname.lower() != (urlsplit(self.final_url).hostname or "").lower(): self.external_form = True
        if tag == "a" and a.get("href") is not None:
            href = a["href"].strip()
            if not href or href == "#": self.empty_refs += 1
            elif href.lower().startswith(("javascript:", "mailto:", "tel:", "data:")): pass
            else:
                try:
                    target = urlsplit(urljoin(self.final_url, href))
                except ValueError:
                    return
                if target.hostname and target.hostname.lower() == (urlsplit(self.final_url).hostname or "").lower(): self.self_refs += 1
                elif target.hostname: self.external_refs += 1
                else: self.self_refs += 1

    def handle_endtag(self, tag: str):
        if tag.lower() == "title": self.in_title = False

    def handle_data(self, data: str):
        if self.in_title: self.title_chunks.append(data)


def extract_features(original_url: str, page: PageFetch) -> tuple[dict[str, object], list[str], list[str]]:
    """Extract transparent URL/static-HTML signals; unresolved required features remain missing and prevent model inference."""
    from urllib.parse import urlsplit
    supplied = original_url.strip()
    if not re.match(r"^https?://", supplied, flags=re.IGNORECASE):
        supplied = f"https://{supplied}"
    # Use the same validated canonical URL for URL text, lengths, counts, and page retrieval.
    raw = _normalize_url(supplied)
    normalized = raw
    parts = urlsplit(normalized)
    host = parts.hostname or ""
    try:
        ip = ipaddress.ip_address(host)
        is_ip = 1
        domain = host
        suffix = ""
        subdomains = 0
    except ValueError:
        is_ip = 0
        domain = host
        labels = [label for label in host.split(".") if label]
        # The supplied CSV's Domain is the full hostname and TLD is its final
        # label (for example, www.example.co.uk -> Domain=www.example.co.uk,
        # TLD=uk, NoOfSubDomain=2), rather than the PSL registrable domain.
        suffix = labels[-1] if labels else ""
        subdomains = max(len(labels) - 2, 0)
    length = len(raw)
    letters = sum(c.isalpha() for c in raw)
    digits = sum(c.isdigit() for c in raw)
    special_count = sum(not c.isalnum() for c in raw)
    # Values intentionally left None: their training formulas/data sources were not present in the supplied artifacts.
    features: dict[str, object] = {key: None for key in FEATURE_ORDER}
    features.update({
        "URL": raw, "URLLength": length, "Domain": domain, "DomainLength": len(domain),
        "IsDomainIP": is_ip, "TLD": suffix if suffix else None, "TLDLength": len(suffix) if suffix else None,
        "NoOfSubDomain": subdomains, "NoOfLettersInURL": letters,
        "LetterRatioInURL": letters / length if length else None,
        "NoOfDegitsInURL": digits, "DegitRatioInURL": digits / length if length else None,
        "NoOfEqualsInURL": raw.count("="), "NoOfQMarkInURL": raw.count("?"),
        "NoOfAmpersandInURL": raw.count("&"), "SpacialCharRatioInURL": special_count / length if length else None,
        "IsHTTPS": int(parts.scheme == "https"),
        "NoOfURLRedirect": page.redirects, "NoOfSelfRedirect": page.self_redirects,
    })
    warnings = list(page.warnings)
    missing = [k for k, v in features.items() if v is None]
    if not page.html:
        warnings.append("No static HTML was available; webpage-dependent features were left missing.")
    else:
        parser = _PageSignals(page.final_url)
        try:
            parser.feed(page.html)
            parser.close()
        except (ValueError, AssertionError):
            warnings.append("The HTML was malformed and could not be parsed completely; uncertain webpage features remain unavailable.")
        title = " ".join(" ".join(parser.title_chunks).split())
        lines = page.html.splitlines()
        features.update({
            "LineOfCode": len(lines), "LargestLineLength": max((len(x) for x in lines), default=0),
            "HasTitle": int(parser.has_title), "Title": title or None,
            "HasFavicon": int(parser.favicon) if parser.favicon is not None else None,
            "IsResponsive": int(parser.responsive) if parser.responsive is not None else None,
            "HasDescription": int(parser.has_description), "NoOfiFrame": parser.iframes,
            "HasExternalFormSubmit": int(parser.external_form), "HasSubmitButton": int(parser.has_submit),
            "HasHiddenFields": int(parser.has_hidden), "HasPasswordField": int(parser.has_password),
            "NoOfImage": parser.images, "NoOfCSS": parser.css, "NoOfJS": parser.js,
            "NoOfSelfRef": parser.self_refs, "NoOfEmptyRef": parser.empty_refs,
            "NoOfExternalRef": parser.external_refs,
        })
        missing = [k for k, v in features.items() if v is None]
    warnings.append("Dataset-specific formulas or external evidence were not supplied for some features; these fields remain missing. The trained pipeline contains fitted imputers, but incomplete analyses are not passed to it.")
    warnings.append("The supplied training metadata reports no missing values. The model runs only when all 54 required features are available; missing values in an incomplete analysis are not imputed.")
    warnings.append("Only static HTML is inspected. Website JavaScript is never executed, so dynamic signals may be absent.")
    if page.redirects:
        warnings.append("Redirect counts reflect the safe static fetch chain only; client-side redirects are not executed.")
    return features, sorted(set(missing)), warnings
