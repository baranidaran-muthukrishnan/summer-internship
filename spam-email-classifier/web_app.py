"""Small local web interface for the spam email classifier.

Run ``python web_app.py`` and open http://localhost:8000 in a browser.
Train the model first with ``python train.py`` if outputs/spam_model.pkl is absent.
"""

from __future__ import annotations

from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from src.spam_classifier import MultinomialNaiveBayes


MODEL_PATH = Path("outputs/spam_model.pkl")


def page(content: str) -> bytes:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spam Email Classifier</title>
<style>
body {{ font-family: Arial, sans-serif; background:#f4f7fb; color:#172033; margin:0; }}
main {{ max-width:760px; margin:48px auto; background:#fff; padding:36px; border-radius:14px; box-shadow:0 6px 24px #17203318; }}
h1 {{ margin-top:0; color:#17375e; }} p {{ line-height:1.5; color:#526174; }}
textarea {{ width:100%; min-height:180px; box-sizing:border-box; padding:12px; border:1px solid #b9c5d4; border-radius:8px; font:inherit; }}
button {{ margin-top:16px; border:0; border-radius:8px; padding:12px 18px; background:#28649a; color:#fff; font-weight:bold; cursor:pointer; }}
.result {{ margin-top:25px; padding:18px; border-radius:8px; background:#edf6f2; }} .spam {{ background:#fff0ef; }}
.score {{ font-size:24px; font-weight:bold; }}
</style></head><body><main>{content}</main></body></html>""".encode("utf-8")


def form(message: str = "", result: str = "") -> str:
    return f"""
<h1>Spam Email Classifier</h1>
<p>Paste an email or SMS message. The local model predicts whether it is spam or ham.</p>
<form method="post"><textarea name="message" placeholder="Enter a message...">{escape(message)}</textarea><br><button type="submit">Classify message</button></form>
{result}
"""


class ClassifierHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(page(form()))

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        values = parse_qs(self.rfile.read(length).decode("utf-8"))
        message = values.get("message", [""])[0].strip()
        if not message:
            result = '<div class="result spam">Please enter a message.</div>'
        elif not MODEL_PATH.exists():
            result = '<div class="result spam">Model not found. Run <code>python train.py</code> first.</div>'
        else:
            model = MultinomialNaiveBayes.load(MODEL_PATH)
            probabilities = model.predict_proba_one(message)
            label = max(probabilities, key=probabilities.get)
            css = "spam" if label == "spam" else ""
            result = (
                f'<div class="result {css}"><div class="score">Prediction: {escape(label.title())}</div>'
                f'<p>Spam probability: {probabilities.get("spam", 0):.1%}<br>'
                f'Ham probability: {probabilities.get("ham", 0):.1%}</p></div>'
            )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(page(form(message, result)))


if __name__ == "__main__":
    print("Open http://localhost:8000 in your browser. Press Ctrl+C to stop.")
    HTTPServer(("localhost", 8000), ClassifierHandler).serve_forever()
