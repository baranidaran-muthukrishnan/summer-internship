# Customer Service Chatbot

A self-contained rule-based chatbot for answering common customer service questions.

## How to run

Open `index.html` in a browser. No build step or server is required.

## What it demonstrates

- Intent design for common customer support categories
- Text normalization and lightweight stemming
- Keyword and example-based scoring
- Fallback handling when confidence is low
- Quick replies based on the detected intent

## Included intents

- Order tracking
- Refunds
- Returns
- Shipping
- Payments
- Account access
- Store hours / live support
- Greetings

## Suggested improvements

- Store chat history in local storage
- Add a support ticket form for fallback cases
- Connect intents to a real FAQ database
- Replace rule-based scoring with a trained classifier
