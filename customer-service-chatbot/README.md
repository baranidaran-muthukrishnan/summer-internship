# Customer Service Chatbot

A self-contained, browser-based customer-service chatbot for common support questions.

## How to run

Open `index.html` in a browser. No build step or server is required.

## Included support topics

- Order tracking
- Refunds
- Returns
- Shipping
- Payments
- Account access
- Store hours / live support
- Greetings

The chatbot checks direct phrases first, then support keywords. If it cannot identify a topic, it asks the user to choose a clearer question instead of giving an unrelated answer.

## Suggested improvements

- Store chat history in local storage
- Add a support ticket form for fallback cases
- Connect intents to a real FAQ database
- Replace rule-based scoring with a trained classifier
