const knowledgeBase = [
  { topic:"Order tracking", phrases:["track an order","track my order","where is my order","where is my package","order status"], words:["tracking","track","order","package","parcel","delayed","delivery"], answer:"Open Account > Orders and select the order to view its delivery status. If there is no update for two business days, contact support with your order number." },
  { topic:"Refunds", phrases:["request a refund","refund status","where is my refund","money back"], words:["refund","refunded","reimbursement","cashback"], answer:"Refunds are issued after the returned item is approved. The amount normally appears in 5 to 7 business days, depending on your bank." },
  { topic:"Returns", phrases:["start a return","return an item","exchange an item","damaged item"], words:["return","exchange","replace","damaged","defective","wrong"], answer:"Go to Account > Orders, select the item, and choose Return or Exchange. You can then download the return instructions." },
  { topic:"Payments", phrases:["payment failed","card declined","charged twice","payment issue"], words:["payment","card","billing","charged","charge","declined","checkout"], answer:"Check that your card number, billing address, and bank approval are correct. A duplicate pending charge should disappear within 24 to 48 hours." },
  { topic:"Account access", phrases:["reset my password","forgot my password","cannot log in","change my email"], words:["password","login","account","reset","locked","email"], answer:"Select Forgot password on the sign-in screen to receive a reset link. For an email change, contact support so ownership can be verified." },
  { topic:"Human support", phrases:["talk to support","talk to an agent","talk to a person","contact support"], words:["agent","human","support","representative","contact","hours"], answer:"A support agent is available Monday to Friday from 9 AM to 6 PM. Please include your order number and a short description of the issue." },
];

const chat = document.querySelector("#chat");
const form = document.querySelector("#chat-form");
const input = document.querySelector("#message");
const suggestions = document.querySelector("#suggestions");

function clean(text) { return text.toLowerCase().replace(/[^a-z0-9\s]/g, " ").replace(/\s+/g, " ").trim(); }
function findAnswer(message) {
  const text = clean(message); const tokens = new Set(text.split(" ")); let winner = null;
  for (const item of knowledgeBase) {
    const phraseScore = item.phrases.filter((phrase) => text.includes(phrase)).length * 8;
    const wordScore = item.words.filter((word) => tokens.has(word)).length * 2;
    const score = phraseScore + wordScore;
    if (!winner || score > winner.score) winner = { item, score };
  }
  return winner && winner.score >= 2 ? winner.item : null;
}
function addMessage(text, role, hint = "") {
  const bubble = document.createElement("div");
  bubble.className = "message " + role;
  bubble.textContent = text;
  if (hint) { const label = document.createElement("div"); label.className = "hint"; label.textContent = hint; bubble.append(label); }
  chat.append(bubble); chat.scrollTop = chat.scrollHeight;
}
function respond(message) {
  const answer = findAnswer(message);
  if (answer) addMessage(answer.answer, "bot", answer.topic);
  else addMessage("I did not understand that request. Please ask about an order, return, refund, payment, password, or speaking to support.", "bot", "Try a suggested question");
}
function send(text) {
  const message = text.trim(); if (!message) return;
  addMessage(message, "user"); input.value = "";
  window.setTimeout(() => respond(message), 160);
}
form.addEventListener("submit", (event) => { event.preventDefault(); send(input.value); });
suggestions.addEventListener("click", (event) => { if (event.target.matches("button")) send(event.target.textContent); });
document.querySelector("#restart").addEventListener("click", () => { chat.replaceChildren(); addMessage("Welcome! Ask a question or choose a suggested topic.", "bot", "Orbit Support"); input.focus(); });
document.querySelector("#restart").click();
