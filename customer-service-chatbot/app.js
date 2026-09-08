const intents = [
  {
    name: "Order tracking",
    examples: ["track order", "where is my package", "shipping status", "order status", "delivery update"],
    keywords: ["track", "tracking", "order", "package", "parcel", "delivery", "shipped", "status"],
    response:
      "You can track your order from Account > Orders. If you share your order ID with a support agent, they can check the latest carrier scan for you.",
    followUps: ["Where is my package?", "Change delivery address"],
  },
  {
    name: "Refunds",
    examples: ["refund status", "where is my refund", "money back", "return refund"],
    keywords: ["refund", "money", "credited", "reimbursement", "cashback"],
    response:
      "Refunds are usually processed within 5 to 7 business days after the returned item is inspected. Bank processing can add another 2 business days.",
    followUps: ["Start a return", "Refund status"],
  },
  {
    name: "Returns",
    examples: ["return item", "start return", "exchange product", "wrong item"],
    keywords: ["return", "exchange", "replace", "wrong", "damaged", "defective"],
    response:
      "Most items can be returned within 30 days in unused condition. Go to Account > Orders, choose the item, and select Return or Exchange.",
    followUps: ["Return policy", "Damaged item"],
  },
  {
    name: "Shipping",
    examples: ["shipping cost", "delivery time", "free shipping", "international shipping"],
    keywords: ["shipping", "delivery", "ship", "cost", "fee", "international", "express", "standard"],
    response:
      "Standard shipping takes 3 to 5 business days. Express shipping takes 1 to 2 business days. Free standard shipping applies to eligible orders over $50.",
    followUps: ["Shipping cost", "Express delivery"],
  },
  {
    name: "Payments",
    examples: ["payment failed", "card declined", "billing issue", "charged twice"],
    keywords: ["payment", "card", "billing", "charged", "declined", "invoice", "paid", "checkout"],
    response:
      "For payment issues, confirm your card details, billing address, and bank approval. Duplicate pending charges normally disappear within 24 to 48 hours.",
    followUps: ["Card declined", "Charged twice"],
  },
  {
    name: "Account access",
    examples: ["reset password", "cannot login", "forgot password", "change email"],
    keywords: ["login", "password", "account", "email", "sign", "reset", "locked"],
    response:
      "Use Forgot password on the sign-in page to reset access. If your email changed, contact support so they can verify ownership before updating it.",
    followUps: ["Reset password", "Change email"],
  },
  {
    name: "Store hours",
    examples: ["support hours", "contact support", "talk to agent", "business hours"],
    keywords: ["hours", "agent", "support", "contact", "representative", "human", "call"],
    response:
      "Live support is available Monday to Friday, 9 AM to 6 PM. You can still leave a message anytime and the team will reply by the next business day.",
    followUps: ["Talk to an agent", "Support hours"],
  },
  {
    name: "Greeting",
    examples: ["hello", "hi", "hey", "good morning"],
    keywords: ["hello", "hi", "hey", "morning", "evening"],
    response: "Hi! I can help with orders, returns, refunds, shipping, payments, and account access.",
    followUps: ["Track my order", "Start a return"],
  },
];

const fallbackResponse =
  "I am not fully sure about that yet. Try asking about orders, returns, refunds, shipping, payments, or account access. For complex issues, I can route you to a human agent.";

const starterPrompts = [
  "Track my order",
  "Start a return",
  "Where is my refund?",
  "Payment failed",
  "Reset password",
];

const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chatForm");
const inputEl = document.querySelector("#messageInput");
const quickRepliesEl = document.querySelector("#quickReplies");
const resetEl = document.querySelector("#resetChat");
const intentCountEl = document.querySelector("#intentCount");

intentCountEl.textContent = intents.length.toString();

function normalize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean)
    .map(stemWord);
}

function stemWord(word) {
  return word
    .replace(/ies$/, "y")
    .replace(/ing$/, "")
    .replace(/ed$/, "")
    .replace(/s$/, "");
}

function scoreIntent(tokens, intent) {
  const keywordSet = new Set(intent.keywords.map(stemWord));
  const exampleTokens = normalize(intent.examples.join(" "));
  let score = 0;

  for (const token of tokens) {
    if (keywordSet.has(token)) score += 3;
    if (exampleTokens.includes(token)) score += 1;
  }

  return score / Math.max(tokens.length, 1);
}

function findBestIntent(message) {
  const tokens = normalize(message);
  const ranked = intents
    .map((intent) => ({ intent, score: scoreIntent(tokens, intent) }))
    .sort((a, b) => b.score - a.score);

  return ranked[0].score >= 1.1 ? ranked[0] : null;
}

function addMessage(role, text, meta = "") {
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  const metaEl = document.createElement("div");
  metaEl.className = "meta";
  metaEl.textContent = meta || (role === "bot" ? "HelpDesk Bot" : "You");

  wrapper.append(bubble, metaEl);
  messagesEl.append(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderQuickReplies(prompts) {
  quickRepliesEl.replaceChildren();
  prompts.forEach((prompt) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = prompt;
    button.addEventListener("click", () => submitMessage(prompt));
    quickRepliesEl.append(button);
  });
}

function botReply(message) {
  const match = findBestIntent(message);
  const typingDelay = 250 + Math.min(message.length * 8, 450);

  window.setTimeout(() => {
    if (!match) {
      addMessage("bot", fallbackResponse, "Fallback response");
      renderQuickReplies(starterPrompts);
      return;
    }

    addMessage("bot", match.intent.response, `Matched: ${match.intent.name}`);
    renderQuickReplies(match.intent.followUps);
  }, typingDelay);
}

function submitMessage(message) {
  const cleaned = message.trim();
  if (!cleaned) return;

  addMessage("user", cleaned);
  inputEl.value = "";
  botReply(cleaned);
}

function resetChat() {
  messagesEl.replaceChildren();
  addMessage(
    "bot",
    "Hello! Ask me a customer service question, or choose one of the suggested prompts.",
    "HelpDesk Bot"
  );
  renderQuickReplies(starterPrompts);
  inputEl.focus();
}

formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  submitMessage(inputEl.value);
});

resetEl.addEventListener("click", resetChat);

resetChat();
