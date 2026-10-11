"""Loopback integration checks. No SSH worker or physical robot is started."""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest
import local_dashboard as app

CALLS = []


class FakeJetson(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == '/':
            data, code = b'<meta name="a1z-token" content="abcd1234">', 200
        elif self.path == '/api/video/demo' and self.headers.get('Range') == 'bytes=2-4':
            data, code = b'234', 206
        else:
            data, code = b'{"state":"await_ready"}', 200
        self.send_response(code)
        self.send_header('Content-Length', str(len(data)))
        if code == 206:
            self.send_header('Content-Range', 'bytes 2-4/10')
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        CALLS.append((self.path, body, self.headers.get('X-A1Z-Token')))
        data = b'{"ok":true}'
        self.send_response(200)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class DashboardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.remote = ThreadingHTTPServer(('127.0.0.1', 0), FakeJetson)
        cls.local = ThreadingHTTPServer(('127.0.0.1', 0), app.Handler)
        app.PORT = cls.local.server_port
        app.UPSTREAM_PORT = cls.remote.server_port
        for server in (cls.remote, cls.local):
            threading.Thread(target=server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        for server in (cls.local, cls.remote):
            server.shutdown()
            server.server_close()

    def setUp(self):
        CALLS.clear()
        app.ONLINE.clear()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', app.PORT, timeout=2)
        try:
            conn.request(method, path, body, headers or {})
            r = conn.getresponse()
            return r.status, dict(r.getheaders()), r.read()
        finally:
            conn.close()

    def test_offline_page_and_no_command_replay(self):
        status, _, body = self.request('GET', '/')
        self.assertEqual(status, 200)
        self.assertIn(b'A1Z Console', body)
        self.assertNotIn(b'__TOKEN__', body)
        self.assertEqual(self.request('GET', '/static/app.js')[0], 200)
        self.assertEqual(self.request('GET', '/api/session')[0], 503)
        self.assertEqual(self.request('POST', '/api/session/key', b'{"key":"READY"}',
            {'X-A1Z-Token': app.TOKEN})[0], 503)
        app.ONLINE.set()
        self.assertEqual(self.request('GET', '/api/session')[0], 200)
        self.assertEqual(CALLS, [])

    def test_authenticated_forwarding_exactly_once(self):
        app.ONLINE.set()
        body = b'{"key":"TEST_ONLY"}'
        status, _, result = self.request('POST', '/api/session/key', body,
            {'X-A1Z-Token': app.TOKEN, 'Origin': f'http://localhost:{app.PORT}'})
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(result)['ok'])
        self.assertEqual(CALLS, [('/api/session/key', body, 'abcd1234')])

    def test_access_boundaries(self):
        app.ONLINE.set()
        self.assertEqual(self.request('GET', '/', headers={'Host': 'untrusted.example'})[0], 403)
        self.assertEqual(self.request('POST', '/api/session/key', b'{}')[0], 403)
        self.assertEqual(self.request('POST', '/api/session/key', b'{}',
            {'X-A1Z-Token': app.TOKEN, 'Origin': 'https://untrusted.example'})[0], 403)
        self.assertEqual(self.request('GET', '/static/%2e%2e/launcher/local_dashboard.py')[0], 404)
        self.assertEqual(CALLS, [])

    def test_video_range_preserved(self):
        app.ONLINE.set()
        code, headers, data = self.request('GET', '/api/video/demo', headers={'Range': 'bytes=2-4'})
        self.assertEqual((code, headers['Content-Range'], data), (206, 'bytes 2-4/10', b'234'))


if __name__ == '__main__':
    unittest.main()
