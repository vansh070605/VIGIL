from http.server import SimpleHTTPRequestHandler
import socketserver

PORT = 8000

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super().end_headers()

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"Serving at port {PORT}")
        httpd.serve_forever()
