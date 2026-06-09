# agent_graph.py

from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
from typing_extensions import NotRequired

# Shared state schema
class GraphState(TypedDict):
    input: str
    route: NotRequired[str]
    result: NotRequired[str]

# Router node
def router_node(state: Annotated[GraphState, "input"]) -> Annotated[GraphState, "route"]:
    prompt = state["input"].lower()
    if "summarize" in prompt:
        print("[Router Node] -> [Summarizer Node] -> ", end="")
        return {"route": "summarizer"}
    elif "translate" in prompt:
        print("[Router Node] -> [Translator Node] -> ", end="")
        return {"route": "translator"}
    elif any(op in prompt for op in ["+", "-", "*", "/"]):
        print("[Router Node] -> [Math Solver Node] -> ", end="")
        return {"route": "math"}
    else:
        print("[Router Node] -> [Fallback Node] -> ", end="")
        return {"route": "fallback"}

# Math node
def math_node(state: Annotated[GraphState, "input"]) -> Annotated[GraphState, "result"]:
    try:
        result = eval(state["input"])
        return {"result": f"✅ Result: {result}"}
    except Exception as e:
        return {"result": f"❌ Error: {str(e)}"}

# Summarizer node
def summary_node(state: Annotated[GraphState, "input"]) -> Annotated[GraphState, "result"]:
    llm = OllamaLLM(model="mistral")
    prompt = f"Summarize this: {state['input']}"
    summary = llm.invoke(prompt)
    return {"result": summary}

# Translator node
def translator_node(state: Annotated[GraphState, "input"]) -> Annotated[GraphState, "result"]:
    llm = OllamaLLM(model="mistral")
    prompt = f"Translate this to another language: {state['input']}"
    translated = llm.invoke(prompt)
    return {"result": translated}

# Fallback node
def fallback_node(state: Annotated[GraphState, "input"]) -> Annotated[GraphState, "result"]:
    return {"result": "🤔 Sorry, I can only handle math, summarize, or translate requests."}

# Optional printer node (for logging)
def printer_node(state: Annotated[GraphState, "result"]) -> None:
    print("[Printer Node]")
    print("Final Output:", state["result"])

# Build graph
graph = StateGraph(GraphState)
graph.add_node("router", router_node)
graph.add_node("math", math_node)
graph.add_node("summarizer", summary_node)
graph.add_node("translator", translator_node)
graph.add_node("fallback", fallback_node)
graph.add_node("final", printer_node)

graph.set_entry_point("router")
graph.add_conditional_edges("router", lambda state: state["route"], {
    "math": "math",
    "summarizer": "summarizer",
    "translator": "translator",
    "fallback": "fallback"
})
graph.add_edge("math", "final")
graph.add_edge("summarizer", "final")
graph.add_edge("translator", "final")
graph.add_edge("fallback", "final")

app = graph.compile()

# Async function to invoke
async def process_user_input(user_prompt: str) -> str:
    state = {"input": user_prompt}
    final_state = await app.ainvoke(state)
    return final_state.get("result", "⚠️ No output")
