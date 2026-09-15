"""Static server for web/ that sends charset=utf-8 (python's default does not)."""
import functools, http.server, socketserver
class H(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      '.html': 'text/html; charset=utf-8',
                      '.js': 'text/javascript; charset=utf-8',
                      '.json': 'application/json; charset=utf-8'}
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(('', 8777), functools.partial(H, directory='web')) as s:
    s.serve_forever()
