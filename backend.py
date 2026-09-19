from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import Tool, tool
from dotenv import load_dotenv
from pypdf import PdfReader
import sqlite3
import requests
import io
import contextvars
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper


load_dotenv()

# RAG embeddings
_embeddings = None
_rag_available = False
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    _rag_available = True
except Exception:
    _rag_available = False

# Context var for current thread_id (set by frontend before streaming so RAG tool can use it)
current_thread_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("thread_id", default=None)


### ********************TOOLS DEFINITION SECTION******************** ###
search_tool = DuckDuckGoSearchRun(region="in", safesearch="Moderate")
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()) #wikipwdia tool
wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())




@tool
def calculator(first_num:float, second_num:float, operation:str) -> dict:
    """Here is a simple calculator which can perform basic arithmatic operations but keep in mind to follow BODMAS rules"""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed."}
            result = first_num / second_num
        else:
            return {"error": "Invalid operation. Please use add, sub, mul, or div."}
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """Get the current stock price for a given symbol using a public API. If it is not able to fetch then use duckduckgosearch"""
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey=55W3PSBIJOH81THN'
    r = requests.get(url)
    return r.json()


### ******************** RAG (PDF) SECTION ******************** ###
# Per-thread in-memory vector stores for uploaded PDFs
_rag_vector_stores: dict[str, FAISS] = {}
_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
_RAG_DEPS_MSG = "PDF Q&A is unavailable: dependency error (sentence-transformers/huggingface_hub). Run: pip install -U huggingface_hub transformers sentence-transformers"


def _load_pdf_documents(pdf_bytes: bytes) -> list[Document]:
    """Load PDF from bytes using pypdf and return LangChain Documents. use it to answer from PDF uploaded by the user."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            docs.append(Document(page_content=text.strip(), metadata={"page": i + 1, "source": "uploaded_pdf"}))
    return docs


def add_pdf_for_thread(thread_id: str, pdf_bytes: bytes) -> str:
    """Ingest an uploaded PDF for the given thread. Creates/updates the vector store for RAG."""
    if not _rag_available or _embeddings is None:
        return _RAG_DEPS_MSG
    docs = _load_pdf_documents(pdf_bytes)
    if not docs:
        return "No text could be extracted from the PDF."
    chunks = _text_splitter.split_documents(docs)
    if thread_id in _rag_vector_stores:
        del _rag_vector_stores[thread_id]
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=_embeddings,
    )
    _rag_vector_stores[thread_id] = vector_store
    return f"PDF ingested successfully. {len(chunks)} chunks from {len(docs)} page(s). You can now ask questions about this document."


@tool
def query_uploaded_document(question: str) -> str:
    """Search the PDF document that the user uploaded in this chat and answer questions from it.
    Use this when the user asks about their uploaded PDF, e.g. 'what is in my document?', 'summarize the PDF', 'what does the document say about X?'.
    Returns relevant excerpts from the document. If no PDF was uploaded for this chat, returns a message saying so."""
    if not _rag_available:
        return _RAG_DEPS_MSG
    thread_id = current_thread_id_ctx.get()
    if not thread_id or thread_id not in _rag_vector_stores:
        return "No PDF has been uploaded in this chat. Ask the user to upload a PDF first, then they can ask questions about it."
    vector_store = _rag_vector_stores[thread_id]
    results = vector_store.similarity_search(question, k=4)
    if not results:
        return "No relevant passages found in the uploaded document for this question."
    excerpts = "\n\n---\n\n".join(doc.page_content for doc in results)
    return f"Relevant excerpts from the uploaded PDF:\n\n{excerpts}"


tools = [search_tool, calculator, get_stock_price, query_uploaded_document, wikipedia,wikidata]



llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature = 0.6
)

llm = llm.bind_tools(tools)

# System prompt for the chatbot
SYSTEM_PROMPT = """
You are a helpful chatbot agent and your name is Srii your are transformed by Deepanshu Bairagi.
Answer users question interectivly.
Here are some points to keep in mind while answering:
1. First think that whether you can use any tools to answer or not. If tool is clearly not needed the answer by yourself.
2. If one tool fails to answer then try another tool just like of duckduckgo not work use wikipedia or for better result use both simultaneously.
3. Make sure to use BODMAS while using calculator.
4. You can use multiple tool at once if needed.
5. If user has uploaded a PDF then try to use informations of that PDF.
6. If user ask any technical or study related question after answering ask him 2 related questions for follow up.
7. try to answer in points and organised way
"""

#state definition
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """LLM node that may answer the question directly and can call tools if needed."""
    messages = state["messages"]
    
    # Prepend system prompt if it's not already the first message
    if not messages or not isinstance(messages[0], SystemMessage):
        messages_with_system = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    else:
        messages_with_system = messages

    try:
        response = llm.invoke(messages_with_system)
        return {"messages": response}
    except Exception as e:
        # Groq can raise when the model outputs invalid tool calls; return a fallback so the chat doesn't crash
        err_msg = str(e).split("failed_generation")[0].strip() if "failed_generation" in str(e) else str(e)
        return {"messages": [AIMessage(content=f"I ran into an issue while handling that: {err_msg}. Please try rephrasing or asking something else.")]}

tool_node = ToolNode(tools)

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


### ******************Graph definition ****************** ###
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)


graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def rerieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
