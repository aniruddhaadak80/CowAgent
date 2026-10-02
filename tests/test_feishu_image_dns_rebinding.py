# encoding:utf-8
"""The image SSRF guard must judge the address the request actually reaches.

``validate_public_image_url`` resolves the hostname and checks that every
address it got back is public. ``requests`` then resolves the same hostname
again, on its own, when it opens the socket. A DNS name under attacker control
with a short TTL can answer the first lookup with a public address and the
second with 127.0.0.1 -- so the guard's verdict is already stale by the time the
connection is made, and bytes from an internal address are read and uploaded to
Feishu.

The check and the connection now share one resolution: the address that passed
is the address the pool is opened against, with ``Host`` and SNI still carrying
the original name so certificate verification is unchanged.

The harness below runs a real HTTP server on loopback standing in for the
internal service, and hands out the public address on the first lookup and the
loopback address on every later one -- exactly what a rebinding record does. A
rebind that reaches the server is the bug, so the assertions count the requests
the server actually received.
"""

import http.server
import os
import socket
import sys
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from channel.feishu import feishu_static_card

# A routable-looking address that nothing answers on, standing in for the public
# IP the first lookup hands back. The pin means the connection goes here and
# fails; without the pin it would be the second lookup's loopback that connects.
_PUBLIC_IP = "93.184.216.34"


class _RebindingHandler(http.server.BaseHTTPRequestHandler):
    """Serves one image and counts how many requests actually arrived."""

    def do_GET(self):
        self.server.hits.append(self.path)
        body = b"INTERNAL-SECRET-BYTES"
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class ImageDownloadDnsRebindingTest(unittest.TestCase):
    """A hostname that re-points at an internal address must not be followed."""

    def setUp(self):
        self.server = http.server.HTTPServer(("127.0.0.1", 0), _RebindingHandler)
        self.server.hits = []
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop)

        # Lookups are counted: the guard takes the first, requests would take
        # the second. Only the guard's answer is ever meant to be used.
        self.lookups = []
        self._real_getaddrinfo = socket.getaddrinfo

        def rebinding_getaddrinfo(host, port, *args, **kwargs):
            if str(host) in ("rebind.test", "rebind2.test"):
                self.lookups.append(host)
                address = _PUBLIC_IP if len(self.lookups) == 1 else "127.0.0.1"
                return [
                    (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port or 80))
                ]
            return self._real_getaddrinfo(host, port, *args, **kwargs)

        socket.getaddrinfo = rebinding_getaddrinfo
        self.addCleanup(self._restore)

    def _restore(self):
        socket.getaddrinfo = self._real_getaddrinfo

    def _stop(self):
        self.server.shutdown()
        self.server.server_close()

    def _url(self, host="rebind.test"):
        return "http://{}:{}/internal.png".format(host, self.port)

    def test_a_rebound_hostname_never_reaches_the_internal_service(self):
        """The guard passes on the public answer, so only the pin can stop this."""
        try:
            feishu_static_card.download_public_image(self._url())
        except Exception:
            # Failing to reach the pinned public address is the expected
            # outcome; what matters is which address it tried.
            pass

        self.assertEqual(
            self.server.hits,
            [],
            "the request reached an internal address after the guard approved "
            "a public one, so the hostname was resolved a second time",
        )
        self.assertEqual(
            len(self.lookups),
            1,
            "the hostname was resolved {} times; the guard and the connection "
            "must share one answer".format(len(self.lookups)),
        )

    def test_the_guard_returns_the_address_it_checked(self):
        """The validated address has to reach the connection somehow."""
        self.assertEqual(
            feishu_static_card.validate_public_image_url(self._url()), _PUBLIC_IP
        )

    def test_a_redirect_to_a_rebound_host_is_also_pinned(self):
        """Each hop re-validates, so the second hop must pin its own answer."""

        class _Redirect:
            status_code = 302
            headers = {"Location": "http://rebind2.test:{}/again.png".format(
                self.server.server_address[1]
            )}

            def close(self):
                pass

        try:
            feishu_static_card.download_public_image(
                self._url(), get=lambda *args, **kwargs: _Redirect()
            )
        except Exception:
            pass

        self.assertEqual(
            self.server.hits,
            [],
            "the redirect target was fetched without pinning its own "
            "validated address",
        )

    def test_a_genuinely_internal_literal_is_still_refused(self):
        """The guard's own verdict must not have been weakened."""
        with self.assertRaises(ValueError) as caught:
            feishu_static_card.validate_public_image_url(
                "http://127.0.0.1:{}/internal.png".format(self.port)
            )
        self.assertIn("non-public", str(caught.exception))

    def test_pinning_does_not_switch_off_certificate_checks(self):
        """The pin moves the address; the certificate must still be checked.

        Rewriting the URL to an IP literal is the tempting shortcut, and it is
        also how TLS gets broken -- or, worse, how ``verify=False`` sneaks in.
        The connection stays on the URL's name for SNI and verification.
        """
        import requests

        session = feishu_static_card._pinned_session(_PUBLIC_IP, "rebind.test")
        self.addCleanup(session.close)
        adapter = session.get_adapter("https://rebind.test/image.png")

        request = requests.Request("GET", "https://rebind.test/image.png").prepare()
        host_params, pool_kwargs = adapter.build_connection_pool_key_attributes(
            request, True, None
        )

        self.assertEqual(host_params["host"], _PUBLIC_IP)
        self.assertEqual(pool_kwargs["assert_hostname"], "rebind.test")
        self.assertEqual(pool_kwargs["server_hostname"], "rebind.test")
        self.assertNotIn(
            "ssl_context", pool_kwargs, "TLS settings must not be overridden"
        )

    def test_a_plain_http_pool_is_not_handed_tls_settings(self):
        """Every http:// image builds its pool here, and TLS keys would break it.

        ``assert_hostname`` and ``server_hostname`` belong to an HTTPS pool.
        A plain HTTP pool forwards its extra keywords to HTTPConnection, which
        takes neither, so pinning an http:// URL while adding them turns every
        plain image download into a TypeError.
        """
        import requests

        session = feishu_static_card._pinned_session(_PUBLIC_IP, "cdn.example.com")
        self.addCleanup(session.close)
        adapter = session.get_adapter("http://cdn.example.com/image.png")

        request = requests.Request("GET", "http://cdn.example.com/image.png").prepare()
        host_params, pool_kwargs = adapter.build_connection_pool_key_attributes(
            request, True, None
        )

        self.assertEqual(host_params["host"], _PUBLIC_IP, "the pin must still apply")
        self.assertNotIn("assert_hostname", pool_kwargs)
        self.assertNotIn("server_hostname", pool_kwargs)

    def test_the_pre_2_32_2_hook_pins_the_pool_too(self):
        """requests is allowed down to 2.28.2, so the pin cannot have one entry.

        Below 2.32.2 requests asks the adapter for a finished pool through
        ``get_connection`` and never reaches the newer attribute hook. If that
        path is left alone the hostname is simply resolved a second time there
        and the address check passes without ever being used.
        """
        import warnings

        session = feishu_static_card._pinned_session(_PUBLIC_IP, "cdn.example.com")
        self.addCleanup(session.close)
        adapter = session.get_adapter("https://cdn.example.com/image.png")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pool = adapter.get_connection("https://cdn.example.com/image.png", None)

        self.assertEqual(pool.host, _PUBLIC_IP)
        self.assertEqual(pool.assert_hostname, "cdn.example.com")
        self.assertEqual(
            pool.conn_kw.get("server_hostname"),
            "cdn.example.com",
            "SNI must stay the hostname, or certificate checking breaks",
        )

    def test_the_request_still_carries_the_original_host_name(self):
        """A virtual-hosted CDN needs Host, not the pinned address."""
        captured = {}

        class _OK:
            status_code = 200
            headers = {"Content-Type": "image/png"}

            def iter_content(self, chunk_size=8192):
                yield b"bytes"

            def close(self):
                pass

        def fake_get(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs.get("headers")
            return _OK()

        def public_getaddrinfo(host, port, *args, **kwargs):
            if str(host) == "cdn.example.com":
                return [
                    (socket.AF_INET, socket.SOCK_STREAM, 6, "", (_PUBLIC_IP, port or 80))
                ]
            return self._real_getaddrinfo(host, port, *args, **kwargs)

        socket.getaddrinfo = public_getaddrinfo
        feishu_static_card.download_public_image(
            "http://cdn.example.com/image.png", get=fake_get
        )

        self.assertEqual(captured["url"], "http://cdn.example.com/image.png")
        self.assertEqual(captured["headers"]["Host"], "cdn.example.com")


if __name__ == "__main__":
    unittest.main()
