const supportTopics = [
  {
    name: "Order tracking",
    phrases: ["track my order", "where is my order", "where is my package", "order status", "delivery status"],
    keywords: ["track", "order", "package", "parcel", "delivery", "shipped", "dispatch", "late"],
    answer: "You can check an order from Account > Orders. Select the order to see its latest delivery status and tracking link.",
    prompts: ["Where is my package?", "Change delivery address"],
  },
  {
    name: "Returns",
    phrases: ["start a return", "return an item", "return my order", "exchange an item", "wrong item"],
    keywords: ["return", "exchange", "replace", "damaged", "defective", "wrong"],
    answer: "To start a return, open Account > Orders, select the item, then choose Return or Exchange. Keep the item and packaging until the return is approved.",
    prompts: ["Return policy", "My item is damaged"],
  },
  {
    name: "Refunds",
    phrases: ["where is my refund", "refund status", "money back", "refund my order"],
    keywords: ["refund", "refunded", "reimbursement", "cashback", "credited"],
    answer: "After a return is approved, refunds normally take 5 to 7 business days. Your bank may take up to 2 additional business days to show the amount.",
    prompts: ["Start a return", "Payment issue"],
  },
  {
    name: "Shipping",
    phrases: ["shipping cost", "shipping fee", "delivery time", "free shipping", "international shipping"],
    keywords: ["shipping", "ship", "delivery", "express", "standard", "international", "fee", "cost"],
    answer: "Standard shipping takes 3 to 5 business days and express shipping takes 1 to 2 business days. Eligible orders over $50 receive free standard shipping.",
    prompts: ["Track my order", "Change delivery address"],
  },
  {
    name: "Payments",
    phrases: ["payment failed", "card declined", "charged twice", "billing issue", "duplicate charge"],
    keywords: ["payment", "card", "billing", "charged", "charge", "declined", "checkout", "invoice"],
    answer: "Please confirm your card details, billing address, and bank approval. If you see a duplicate pending charge, it usually disappears within 24 to 48 hours.",
    prompts: ["Card declined", "Refund status"],
  },
  {
    name: "Account access",
    phrases: ["forgot my password", "reset my password", "cannot log in", "can't log in", "change my email"],
    keywords: ["password", "login", "log", "account", "sign", "reset", "locked", "email"],
    answer: "Use Forgot password on the sign-in page to reset access. For an email-address change, contact support so the team can verify account ownership.",
    prompts: ["Reset password", "Talk to support"],
  },
  {
    name: "Support hours",
    phrases: ["talk to an agent", "talk to a person", "contact support", "support hours", "business hours"],
    keywords: ["agent", "human", "person", "support", "contact", "hours", "representative", "call"],
    answer: "Live support is available Monday to Friday, 9 AM to 6 PM. You can leave a message any time, and the team will reply on the next business day.",
    prompts: ["Support hours", "Track my order"],
  },
];

const greetingWords = new Set(["hello", "hi", "hey", "morning", "afternoon", "evening"]);
const starterPrompts = ["Track my order", "Start a return", "Where is my refund?", "Payment failed", "Reset password"];

const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chatForm");
const inputEl = document.querySelector("#messageInput");
const quickRepliesEl = document.querySelector("#quickReplies");
const resetEl = document.querySelector("#resetChat");
const intentCountEl = document.querySelector("#intentCount");

intentCountEl.textContent = String(supportTopics.length);

function normalize(text) {
  return text.toLowerCase().replace(/[^a-z0-9s']/g, " ").replace(/s+/g, " ").trim();
}

function words(text) {
  return new Set(normalize(text).split(" ").filter(Boolean));
}

function findTopic(message) {
  const text = normalize(message);
  const messageWords = words(message);
  let best = null;

  for (const topic of supportTopics) {
    let score = 0;
    for (const phrase of topic.phrases) {
      if (text.includes(phrase)) score += 10;
    }
    for (const keyword of topic.keywords) {
      if (messageWords.has(keyword)) score += 2;
    }
    if (!best || score > best.score) best = { topic, score };
  }
  return best && best.score >= 2 ? best.topic : null;
}

function responseFor(message) {
  const normalized = normalize(message);
  const topic = findTopic(message);

  if (topic) return { text: topic.answer, meta: topic.name, prompts: topic.prompts };
  if (words(normalized).size <= 4 && [...words(normalized)].some((word) => greetingWords.has(word))) {
    return {
      text: "Hello! I can help with order tracking, returns, refunds, shipping, payments, account access, and support hours.",
      meta: "Greeting",
      prompts: starterPrompts,
    };
  }
  return {
    text: "I could not match that to a support topic. Please try asking about an order, return, refund, shipping, payment, password, or support hours.",
    meta: "Need more detail",
    prompts: starterPrompts,
  };
}

function addMessage(role, text, meta) {
  const message = document.createElement("article");
  message.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  const label = document.createElement("div");
  label.className = "meta";
  label.textContent = meta;
  message.append(bubble, label);
  messagesEl.append(message);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function showPrompts(prompts) {
  quickRepliesEl.replaceChildren();
  prompts.forEach((prompt) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = prompt;
    button.addEventListener("click", () => sendMessage(prompt));
    quickRepliesEl.append(button);
  });
}

function sendMessage(text) {
  const message = text.trim();
  if (!message) return;
  addMessage("user", message, "You");
  inputEl.value = "";
  const reply = responseFor(message);
  window.setTimeout(() => {
    addMessage("bot", reply.text, `HelpDesk Bot · ${reply.meta}`);
    showPrompts(reply.prompts);
  }, 180);
}

function resetChat() {
  messagesEl.replaceChildren();
  addMessage("bot", "Hello! What can I help you with today?", "HelpDesk Bot");
  showPrompts(starterPrompts);
  inputEl.focus();
}

formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(inputEl.value);
});
resetEl.addEventListener("click", resetChat);
resetChat();
