from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

class ThreadingCORSHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

if __name__ == "__main__":
    import os
    import sys

    PORT = 8000
    directory = os.getcwd()
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        os.chdir(directory)

    print(f"🚀 Serving images with CORS at http://localhost:{PORT}/")
    server = ThreadingCORSHTTPServer(('0.0.0.0', PORT), CORSRequestHandler)
    server.serve_forever()
