from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver

PORT = 3000

Handler = SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving at port {PORT}")
    print(f"Open http://localhost:{PORT} in your browser")
    httpd.serve_forever()