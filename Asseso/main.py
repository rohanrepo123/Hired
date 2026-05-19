from typing import Literal

# FastAPI framework
from fastapi import FastAPI

# Used to return custom HTML page
from fastapi.responses import HTMLResponse

# Pydantic models for request/response validation
from pydantic import BaseModel, Field

# Main recommendation pipeline
from recommender import build_reply


# -----------------------------
# FastAPI Application
# -----------------------------

# Create FastAPI app instance
app = FastAPI(
    title="SHL Assessment Recommender"
)


# -----------------------------
# Request / Response Schemas
# -----------------------------

# Single chat message schema
class Message(BaseModel):

    # Message role must be either user or assistant
    role: Literal["user", "assistant"]

    # Message content cannot be empty
    content: str = Field(min_length=1)


# Incoming chat request schema
class ChatRequest(BaseModel):

    # Full conversation history
    messages: list[Message] = Field(min_length=1)


# Individual recommendation schema
class Recommendation(BaseModel):

    # Assessment name
    name: str

    # SHL catalog URL
    url: str

    # Test category/type code
    test_type: str

    # Assessment categories
    keys: str

    # Assessment duration
    duration: str

    # Supported languages
    languages: str


# Final API response schema
class ChatResponse(BaseModel):

    # Assistant reply text
    reply: str

    # Recommended SHL assessments
    recommendations: list[Recommendation]

    # Indicates whether conversation is complete
    end_of_conversation: bool


# -----------------------------
# Frontend Chat UI
# -----------------------------

@app.get("/", response_class=HTMLResponse)
def chat_page() -> str:

    """
    Returns a lightweight browser-based chat interface.
    """

    return """
<!doctype html>
<html lang="en">

<head>

  <!-- Basic page metadata -->
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />

  <!-- Page title -->
  <title>SHL Assessment Chat</title>

  <!-- Inline CSS styling -->
  <style>

    /* Global theme variables */
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1b1f23;
      --muted: #5c6670;
      --line: #d8dee6;
      --accent: #0067b1;
      --accent-dark: #004e86;
      --user: #e8f2ff;
      --assistant: #ffffff;
    }

    * {
      box-sizing: border-box;
    }

    /* Main page layout */
    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--ink);
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }

    /* Header styling */
    header {
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 16px 20px;
    }

    h1 {
      font-size: 20px;
      margin: 0;
      font-weight: 700;
    }

    /* Main chat container */
    main {
      width: min(960px, 100%);
      margin: 0 auto;
      padding: 18px 14px 120px;
    }

    /* Message list layout */
    #messages {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    /* Single message card */
    .message {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      max-width: min(780px, 92%);
      line-height: 1.45;
      white-space: pre-wrap;
    }

    /* User message style */
    .user {
      align-self: flex-end;
      background: var(--user);
      border-color: #b7d8ff;
    }

    /* Assistant message style */
    .assistant {
      align-self: flex-start;
      background: var(--assistant);
    }

    /* Recommendation block */
    .recs {
      margin-top: 10px;
      border-top: 1px solid var(--line);
      padding-top: 8px;
      display: grid;
      gap: 10px;
    }

    /* Recommendation summary */
    .rec-summary {
      color: var(--muted);
      font-size: 13px;
    }

    /* Recommendation table container */
    .rec-table-wrap {
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }

    /* Recommendation table */
    .rec-table {
      width: 100%;
      min-width: 760px;
      border-collapse: collapse;
      font-size: 14px;
      background: #fbfcfe;
    }

    .rec-table th,
    .rec-table td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }

    .rec-table th {
      background: #f2f6fb;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }

    .rec-table tr:last-child td {
      border-bottom: 0;
    }

    /* Recommendation links */
    .rec-table a {
      color: var(--accent-dark);
      font-weight: 700;
      text-decoration: none;
    }

    .rec-table .index {
      width: 36px;
      color: var(--muted);
      white-space: nowrap;
    }

    /* Input form */
    form {
      position: fixed;
      left: 0;
      right: 0;
      bottom: 0;
      background: var(--panel);
      border-top: 1px solid var(--line);
      padding: 12px;
    }

    /* Chat composer layout */
    .composer {
      width: min(960px, 100%);
      margin: 0 auto;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
    }

    /* Input textbox */
    textarea {
      width: 100%;
      min-height: 48px;
      max-height: 140px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      font: inherit;
      line-height: 1.35;
    }

    /* Send button */
    button {
      min-width: 88px;
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: white;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }

    button:disabled {
      opacity: .6;
      cursor: wait;
    }

    /* Mobile responsiveness */
    @media (max-width: 640px) {

      .composer {
        grid-template-columns: 1fr;
      }

      button {
        min-height: 44px;
      }

      .message {
        max-width: 100%;
      }
    }

  </style>
</head>

<body>

  <!-- App header -->
  <header>
    <h1>SHL Assessment Chat</h1>
  </header>

  <!-- Chat messages -->
  <main id="messages"></main>

  <!-- Input form -->
  <form id="chat-form">

    <div class="composer">

      <!-- User input -->
      <textarea
        id="input"
        placeholder="Describe the role..."
        autofocus>
      </textarea>

      <!-- Send button -->
      <button id="send" type="submit">
        Send
      </button>

    </div>

  </form>

  <!-- Frontend JavaScript -->
  <script>

    // Stores complete conversation history
    const messages = [];

    // DOM references
    const list = document.getElementById("messages");
    const form = document.getElementById("chat-form");
    const input = document.getElementById("input");
    const send = document.getElementById("send");

    // Add message to UI
    function addMessage(role, content, recommendations = []) {

      const item = document.createElement("section");

      item.className = `message ${role}`;
      item.textContent = content;

      // Render recommendation table if recommendations exist
      if (recommendations.length) {

        const recs = document.createElement("div");
        recs.className = "recs";

        const summary = document.createElement("div");
        summary.className = "rec-summary";

        summary.textContent =
          `${recommendations.length} SHL assessments matched this role.`;

        recs.append(summary);

        const tableWrap = document.createElement("div");
        tableWrap.className = "rec-table-wrap";

        const table = document.createElement("table");
        table.className = "rec-table";

        const thead = document.createElement("thead");
        const headerRow = document.createElement("tr");

        // Table headers
        ["#", "Name", "Test Type", "Keys", "Duration", "Languages", "URL"]
          .forEach(label => {

            const cell = document.createElement("th");

            cell.textContent = label;

            headerRow.append(cell);
          });

        thead.append(headerRow);

        table.append(thead);

        const tbody = document.createElement("tbody");

        // Render recommendation rows
        recommendations.forEach((rec, idx) => {

          const row = document.createElement("tr");

          const indexCell = document.createElement("td");
          indexCell.textContent = String(idx + 1);
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

      // Append message to page
      list.append(item);

      // Auto scroll
      window.scrollTo({
        top: document.body.scrollHeight,
        behavior: "smooth"
      });
    }

    // Send message to backend API
    async function sendMessage(text) {

      // Store user message
      messages.push({
        role: "user",
        content: text
      });

      // Display user message
      addMessage("user", text);

      send.disabled = true;

      try {

        // API request
        const response = await fetch("/chat", {

          method: "POST",

          headers: {
            "Content-Type": "application/json"
          },

          body: JSON.stringify({
            messages
          })
        });

        // Parse API response
        const data = await response.json();

        // Store assistant response
        messages.push({
          role: "assistant",
          content: data.reply
        });

        // Render assistant response
        addMessage(
          "assistant",
          data.reply,
          data.recommendations || []
        );

      } catch (error) {

        // Fallback UI error
        addMessage(
          "assistant",
          "The local API did not respond. Check the terminal running uvicorn."
        );

      } finally {

        send.disabled = false;

        input.focus();
      }
    }

    // Form submit handler
    form.addEventListener("submit", (event) => {

      event.preventDefault();

      const text = input.value.trim();

      if (!text) return;

      input.value = "";

      sendMessage(text);
    });

    // Enter-to-send support
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


# -----------------------------
# Health Endpoint
# -----------------------------

@app.get("/health")
def health() -> dict[str, str]:

    """
    Health check endpoint required by SHL assignment.
    """

    return {
        "status": "ok"
    }


# -----------------------------
# Main Chat Endpoint
# -----------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> dict:

    """
    Main conversational endpoint.
    
    Receives full conversation history,
    reconstructs conversational context,
    and returns grounded SHL recommendations.
    """

    # Convert Pydantic objects into dictionaries
    messages = [
        msg.model_dump()
        for msg in request.messages
    ]

    # Generate final agent reply
    return build_reply(messages)
