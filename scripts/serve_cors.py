import http.server
import socketserver
import argparse

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super(CORSRequestHandler, self).end_headers()

def serve(port=8000):
    handler = CORSRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving at http://localhost:{port} with CORS enabled")
        httpd.serve_forever()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    serve(args.port)
