"""Helpers for choosing the native Feishu delivery format for text replies."""

import ipaddress
import json
import re
import socket
from contextlib import nullcontext
from typing import Callable, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter

from common.markdown_fence import transform_outside_fences

_BLOCK_MARKDOWN = re.compile(
    r"(?m)^\s{0,3}(?:#{1,6}\s|>\s|[-*+]\s|\d+[.)]\s|```|~~~)"
)
_INLINE_MARKDOWN = re.compile(r"(`[^`\n]+`|\*\*[^*\n]+\*\*|\[[^]\n]+\]\([^)\n]+\))")
_TABLE_SEPARATOR = re.compile(r"(?m)^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_MARKDOWN_IMAGE = re.compile(r"!\[([^\]\n]*)\]\(([^)\s]+)\)")
# An inline code span on one line: a backtick run closed by the next run of the
# same length.
_CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`).*?(?<!`)\1(?!`)")
_REDIRECT_CODES = {301, 302, 303, 307, 308}
_MAX_REDIRECTS = 3
_MAX_REMOTE_IMAGE_BYTES = 10 * 1024 * 1024


def contains_markdown(text: str) -> bool:
    """Return whether *text* contains syntax that benefits from card Markdown."""
    if not text:
        return False
    return bool(
        _BLOCK_MARKDOWN.search(text)
        or _INLINE_MARKDOWN.search(text)
        or _TABLE_SEPARATOR.search(text)
    )


def build_markdown_card(text: str) -> dict:
    """Build an inline Card 2.0 payload with one Markdown element."""
    return {
        "schema": "2.0",
        "config": {},
        "body": {
            "elements": [
                {
                    "tag": "markdown",
                    "content": text,
                }
            ]
        },
    }


def build_text_delivery(text: str) -> Tuple[str, str]:
    """Return the Feishu ``msg_type`` and serialized content for a text reply."""
    if contains_markdown(text):
        return "interactive", json.dumps(build_markdown_card(text), ensure_ascii=False)
    return "text", json.dumps({"text": text}, ensure_ascii=False)


def _outside_code(text: str, transform: Callable[[str], str]) -> str:
    """Apply *transform* to the parts of Markdown *text* outside code blocks
    and inline code spans."""

    def prose(chunk: str) -> str:
        parts = []
        last = 0
        for span in _CODE_SPAN.finditer(chunk):
            parts.append(transform(chunk[last:span.start()]))
            parts.append(span.group(0))
            last = span.end()
        parts.append(transform(chunk[last:]))
        return "".join(parts)

    return transform_outside_fences(text, prose)


def resolve_markdown_images(
    text: str,
    uploader: Callable[[str], Optional[str]],
    max_images: int = 5,
) -> str:
    """Replace remote Markdown image URLs with Feishu image keys.

    Images written inside code blocks or inline code are examples, not images
    to show, so they are left as written and never fetched.
    """
    cache = {}
    uploaded = 0

    def replace(match):
        nonlocal uploaded
        alt = match.group(1).strip() or "image"
        target = match.group(2).strip()
        if target.startswith("img_"):
            return match.group(0)
        if urlparse(target).scheme not in ("http", "https"):
            return match.group(0)

        if target not in cache:
            if uploaded >= max_images:
                cache[target] = None
            else:
                uploaded += 1
                try:
                    cache[target] = uploader(target)
                except Exception:
                    cache[target] = None

        image_key = cache[target]
        if image_key:
            return "![{}]({})".format(alt, image_key)
        return "[Image unavailable: {}]".format(alt)

    return _outside_code(text or "", lambda chunk: _MARKDOWN_IMAGE.sub(replace, chunk))


def validate_public_image_url(url: str) -> str:
    """Reject non-HTTP and non-public image targets, returning the address vetted.

    The address is returned rather than discarded because the caller has to
    connect to *this* answer: a hostname resolves again inside requests, and a
    name with a short TTL can answer the first lookup publicly and the second
    with a loopback or private address, which would leave this verdict stale
    by the time a socket is opened.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("unsupported image URL scheme")
    if not parsed.hostname:
        raise ValueError("image URL has no hostname")

    try:
        literal_address = ipaddress.ip_address(parsed.hostname)
        resolved_addresses = [literal_address]
    except ValueError:
        try:
            addresses = socket.getaddrinfo(
                parsed.hostname,
                parsed.port,
                socket.AF_UNSPEC,
                socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise ValueError("cannot resolve image hostname") from exc
        resolved_addresses = [ipaddress.ip_address(item[4][0]) for item in addresses]

    for address in resolved_addresses:
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise ValueError("image URL resolves to a non-public address")

    # Any address may serve the request, but only one is used, so the check
    # above and the connection below cannot disagree.
    return str(resolved_addresses[0])


class _PinnedHostAdapter(HTTPAdapter):
    """Open connections against one already-validated address.

    ``validate_public_image_url`` resolves the hostname itself and then hands
    the URL to requests, which resolves it a second time when it opens the
    socket. Rebinding the pool's host onto the address that was just checked
    closes that gap: there is no second lookup to rebind. ``assert_hostname``
    and the SNI name stay the original hostname, so the certificate is still
    verified against it -- pinning the address must not quietly weaken TLS,
    which is what rewriting the URL to an IP literal would do.
    """

    def __init__(self, address: str, hostname: str) -> None:
        self._address = address
        self._hostname = hostname
        super().__init__()

    # requests >= 2.32.2 takes the pool apart through this hook and rebuilds it
    # from the returned attributes, so the pin is applied to them.
    def build_connection_pool_key_attributes(self, request, verify, cert):
        host_params, pool_kwargs = super().build_connection_pool_key_attributes(
            request, verify, cert
        )
        host_params["host"] = self._address
        if request.url.lower().startswith("https://"):
            # TLS settings belong to an HTTPS pool alone: handing either of
            # these to a plain HTTP pool reaches HTTPConnection as an
            # unexpected keyword and would break every plain http:// image.
            pool_kwargs["assert_hostname"] = self._hostname
            pool_kwargs["server_hostname"] = self._hostname
        return host_params, pool_kwargs

    # requests < 2.32.2 hands back a finished pool instead, and this project
    # allows requests>=2.28.2, so the pin needs a second entry point as well:
    # without it the override above never runs, the address goes back to being
    # resolved a second time, and the whole check is decorative. The signatures
    # differ between the two requests versions, so this takes *args; the
    # installed version only ever calls one of the two hooks.
    def get_connection(self, *args, **kwargs):
        pool = super().get_connection(*args, **kwargs)
        pool.host = self._address
        if getattr(pool, "scheme", "") == "https":
            pool.assert_hostname = self._hostname
            pool.conn_kw["server_hostname"] = self._hostname
        return pool


def _pinned_session(address: str, hostname: str) -> requests.Session:
    """A session whose every connection goes to ``address``."""
    session = requests.Session()
    adapter = _PinnedHostAdapter(address, hostname)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def download_public_image(
    url: str,
    get=None,
    max_bytes: int = _MAX_REMOTE_IMAGE_BYTES,
) -> Tuple[bytes, str]:
    """Download a public image with redirect, type, and size checks.

    ``get`` is the seam the tests drive; left as None the request goes through
    a session pinned to the address ``validate_public_image_url`` vetted.
    """
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        address = validate_public_image_url(current)
        hostname = urlparse(current).hostname
        # Host keeps naming the CDN rather than the pinned address, which is
        # what a virtual-hosted origin expects to route on.
        headers = {"User-Agent": "CowAgent/Feishu", "Host": hostname}
        pinned = _pinned_session(address, hostname) if get is None else nullcontext()
        with pinned as session:
            fetch = session.get if get is None else get
            response = fetch(
                current,
                headers=headers,
                timeout=(5, 15),
                allow_redirects=False,
                stream=True,
            )

            if response.status_code in _REDIRECT_CODES:
                location = response.headers.get("Location")
                response.close()
                if not location:
                    raise ValueError("image redirect has no location")
                current = urljoin(current, location)
                continue

            if response.status_code != 200:
                response.close()
                raise ValueError("image download returned HTTP {}".format(response.status_code))

            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
            if not content_type.startswith("image/"):
                response.close()
                raise ValueError("remote resource is not an image")

            try:
                content_length = int(response.headers.get("Content-Length") or 0)
            except (TypeError, ValueError):
                content_length = 0
            if content_length > max_bytes:
                response.close()
                raise ValueError("remote image is too large")

            chunks = []
            downloaded = 0
            try:
                for chunk in response.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise ValueError("remote image is too large")
                    chunks.append(chunk)
            finally:
                response.close()
            return b"".join(chunks), content_type

    raise ValueError("too many image redirects")


def upload_public_image_to_feishu(
    url: str,
    access_token: str,
    post=requests.post,
) -> Optional[str]:
    """Download a public image and upload its bytes to Feishu."""
    payload, content_type = download_public_image(url)
    extension = {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
        "image/bmp": "bmp",
    }.get(content_type, "img")
    response = post(
        "https://open.feishu.cn/open-apis/im/v1/images",
        headers={"Authorization": "Bearer " + access_token},
        data={"image_type": "message"},
        files={
            "image": (
                "markdown-image.{}".format(extension),
                payload,
                content_type,
            )
        },
        timeout=(5, 15),
    )
    body = response.json()
    if body.get("code") != 0:
        return None
    return (body.get("data") or {}).get("image_key")
