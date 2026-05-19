from typing import Literal
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from recommender import build_reply

app = FastAPI(title="SHL Assessment Recommender")

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)

class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str
    keys: str
    duration: str
    languages: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool

@app.get("/", response_class=HTMLResponse)
def chat_page() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SHL Assessment Chat</title>
  <style>
    :root { color-scheme: light; --bg: #f6f7f9; --panel: #ffffff; --ink: #1b1f23; --muted: #5c6670; --line: #d8dee6; --accent: #0067b1; --accent-dark: #004e86; --user: #e8f2ff; --assistant: #ffffff; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Arial, Helvetica, sans-serif; background: var(--bg); color: var(--ink); min-height: 100vh; display: grid; grid-template-rows: auto 1fr auto; }
    header { background: var(--panel); border-bottom: 1px solid var(--line); padding: 16px 20px; }
    h1 { font-size: 20px; margin: 0; font-weight: 700; }
    main { width: min(960px, 100%); margin: 0 auto; padding: 18px 14px 120px; }
    #messages { display: flex; flex-direction: column; gap: 12px; }
    .message { border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; max-width: min(780px, 92%); line-height: 1.45; white-space: pre-wrap; }
    .user { align-self: flex-end; background: var(--user); border-color: #b7d8ff; }
    .assistant { align-self: flex-start; background: var(--assistant); }
    .recs { margin-top: 10px; border-top: 1px solid var(--line); padding-top: 8px; display: grid; gap: 10px; }
    .rec-summary { color: var(--muted); font-size: 13px; }
    .rec-table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }
    .rec-table { width: 100%; min-width: 760px; border-collapse: collapse; font-size: 14px; background: #fbfcfe; }
    .rec-table th, .rec-table td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    .rec-table th { background: #f2f6fb; color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; }
    .rec-table tr:last-child td { border-bottom: 0; }
    .rec-table a { color: var(--accent-dark); font-weight: 700; text-decoration: none; }
    .rec-table .index { width: 36px; color: var(--muted); white-space: nowrap; }
    form { position: fixed; left: 0; right: 0; bottom: 0; background: var(--panel); border-top: 1px solid var(--line); padding: 12px; }
    .composer { width: min(960px, 100%); margin: 0 auto; display: grid; grid-template-columns: 1fr auto; gap: 10px; }
    textarea { width: 100%; min-height: 48px; max-height: 140px; resize: vertical; border: 1px solid var(--line); border-radius: 8px; padding: 12px; font: inherit; line-height: 1.35; }
    button { min-width: 88px; border: 0; border-radius: 8px; background: var(--accent); color: white; font: inherit; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: .6; cursor: wait; }
    @media (max-width: 640px) { .composer { grid-template-columns: 1fr; } button { min-height: 44px; } .message { max-width: 100%; } }
  </style>
</head>
<body>
  <header><h1>SHL Assessment Chat</h1></header>
  <main id="messages"></main>
  <form id="chat-form">
    <div class="composer">
      <textarea id="input" placeholder="Describe the role..." autofocus></textarea>
      <button id="send" type="submit">Send</button>
    </div>
  </form>
  <script>
    const messages = [];
    const list = document.getElementById("messages");
    const form = document.getElementById("chat-form");
    const input = document.getElementById("input");
    const send = document.getElementById("send");

    function addMessage(role, content, recommendations = []) {
      const item = document.createElement("section");
      item.className = `message ${role}`;
      item.textContent = content;

      if (recommendations.length) {
        const recs = document.createElement("div");
        recs.className = "recs";
        const summary = document.createElement("div");
        summary.className = "rec-summary";
        summary.textContent = `${recommendations.length} SHL assessments matched this role.`;
        recs.append(summary);
        const tableWrap = document.createElement("div");
        tableWrap.className = "rec-table-wrap";
        const table = document.createElement("table");
        table.className = "rec-table";
        const thead = document.createElement("thead");
        const headerRow = document.createElement("tr");
        ["#", "Name", "Test Type", "Keys", "Duration", "Languages", "URL"].forEach(label => {
          const cell = document.createElement("th");
          cell.textContent = label;
          headerRow.append(cell);
        });
        thead.append(headerRow);
        table.append(thead);
        const tbody = document.createElement("tbody");
        recommendations.forEach((rec, idx) => {
          const row = document.createElement("tr");
          const indexCell = document.createElement("td");
          indexCell.textContent = String(idx+1);
          indexCell.className = "index";
          row.append(indexCell);
          const nameCell = document.createElement("td");
          nameCell.textContent = rec.name;
          row.append(nameCell);
          const typeCell = document.createElement("td");
          typeCell.textContent = rec.test_type;
          row.append(typeCell);
          const keysCell = document.createElement("td");
          keysCell.textContent = rec.keys || "-";
          row.append(keysCell);
          const durationCell = document.createElement("td");
          durationCell.textContent = rec.duration || "-";
          row.append(durationCell);
          const langCell = document.createElement("td");
          langCell.textContent = rec.languages || "-";
          row.append(langCell);
          const linkCell = document.createElement("td");
          const link = document.createElement("a");
          link.href = rec.url;
          link.target = "_blank";
          link.textContent = "Open";
          linkCell.append(link);
          row.append(linkCell);
          tbody.append(row);
        });
        table.append(tbody);
        tableWrap.append(table);
        recs.append(tableWrap);
        item.append(recs);
      }
      list.append(item);
      window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
    }

    async function sendMessage(text) {
      messages.push({ role: "user", content: text });
      addMessage("user", text);
      send.disabled = true;
      try {
        const response = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages })
        });
        const data = await response.json();
        messages.push({ role: "assistant", content: data.reply });
        addMessage("assistant", data.reply, data.recommendations || []);
      } catch (error) {
        addMessage("assistant", "The local API did not respond. Check the terminal running uvicorn.");
      } finally {
        send.disabled = false;
        input.focus();
      }
    }

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      input.value = "";
      sendMessage(text);
    });
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });
  </script>
</body>
</html>
"""

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> dict:
    messages = [msg.model_dump() for msg in request.messages]
    return build_reply(messages)