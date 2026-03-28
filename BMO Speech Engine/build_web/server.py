import http.server
import socketserver
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
class WasmHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".wasm": "application/wasm"}
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        super().end_headers()
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", 8082), WasmHandler) as httpd:
    print("Serving on port 8082", flush=True)
    httpd.serve_forever()
