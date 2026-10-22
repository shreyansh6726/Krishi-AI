import base64
import json
import os
import tempfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from native_language_nlp.groq import create_groq_client, load_dotenv, chat_response


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(PROJECT_DIR, "web")
load_dotenv(os.path.join(PROJECT_DIR, ".env"))


class ChatHandler(SimpleHTTPRequestHandler):
    state = {}
    history = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/chat":
            self._send_json(404, {"error": "Endpoint not found."})
            return

        image_path = None
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))

            message = str(request.get("message", "")).strip()
            if not message:
                raise ValueError("Message cannot be empty.")

            image_path = None
            image_data = request.get("image")
            if image_data:
                _, encoded = image_data.split(",", 1)
                suffix = os.path.splitext(request.get("imageName", ".jpg"))[1] or ".jpg"
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix, dir=os.path.dirname(__file__)
                ) as image_file:
                    image_file.write(base64.b64decode(encoded))
                    image_path = image_file.name

            load_dotenv(os.path.join(PROJECT_DIR, ".env"), override=True)
            use_local = os.getenv("USE_LOCAL_NLP", "false").lower() in ("true", "1", "yes")

            if use_local:
                from native_language_nlp.local_llm import local_chat_response
                reply = local_chat_response(message, self.state, self.history, image_path)
            else:
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    from native_language_nlp.local_llm import local_chat_response
                    reply = local_chat_response(message, self.state, self.history, image_path)
                else:
                    client = create_groq_client(api_key)
                    reply = chat_response(client, message, self.state, self.history, image_path)

            self._send_json(200, {"reply": reply})
        except (ValueError, OSError, RuntimeError, KeyError) as error:
            self._send_json(400, {"error": str(error)})
        except Exception as error:
            if "groq" in getattr(error.__class__, "__module__", ""):
                self._send_json(400, {"error": f"Groq API Error: {error}"})
            else:
                raise
        finally:
            if image_path and os.path.isfile(image_path):
                os.remove(image_path)


def main():
    port = int(os.getenv("KRISHI_AI_PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), ChatHandler)
    print(f"Krishi AI web chatbot: http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
