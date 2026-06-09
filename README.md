# 🤖 LangGraph + Mistral Multi-Agent System

A multi-node agentic AI application built with **LangGraph** and **Mistral 7B** (via Ollama), featuring intelligent routing between specialized agents and a conversational Streamlit interface.

## 🧠 How it works

The system uses a **router node** that reads user intent and dispatches to the right agent automatically:

```
User Input → Router Node → [Summarizer | Translator | Math Solver | Fallback] → Output
```

| Agent | Triggered by | What it does |
|---|---|---|
| 🧮 Math Solver | Arithmetic expressions (`5 + 3`, `12 * 4`) | Evaluates math safely |
| 📝 Summarizer | `summarize: <text>` | Uses Mistral LLM to summarize |
| 🌐 Translator | `translate: <text>` | Uses Mistral LLM to translate |
| 🤔 Fallback | Anything else | Guides the user |

## 🛠️ Tech Stack

- **LangGraph** — stateful multi-node graph orchestration
- **Mistral 7B** — local LLM via Ollama (`langchain_ollama`)
- **Streamlit** — chat UI with session memory
- **Python** — `TypedDict` state schema, async invocation

## 📸 Output Screenshots

### Math Solver
![Math output](output-1.png)

### Summarizer
![Summarizer output](output-2.png)

### Translator
![Translator output](output-3.png)

## 🚀 Run Locally

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com) installed and running with Mistral pulled.

```bash
# Pull Mistral model (one time)
ollama pull mistral

# Install dependencies
pip install langgraph langchain-ollama streamlit

# Run the app
streamlit run app.py
```

## 📁 Project Structure

```
├── agent_graph.py   # LangGraph nodes, routing logic, graph compilation
├── app.py           # Streamlit chat interface
└── output-*.png     # Sample screenshots
```

## 💡 Key Concepts Demonstrated

- **Stateful graphs** using `StateGraph` and `TypedDict` schema
- **Conditional edges** for dynamic routing based on intent
- **Local LLM integration** with Ollama — no API key needed
- **Async agent invocation** with `ainvoke`
- **Session-based chat UI** with Streamlit

---

Built during AI internship at **Ravi Aadhya Infotech** as part of LangGraph + Agentic AI exploration.
