from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)
from langchain_core.documents import Document

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool

from dotenv import load_dotenv
from pypdf import PdfReader

import sqlite3
import requests
import io
import contextvars
import os

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_community.tools.wikidata.tool import (
    WikidataAPIWrapper,
    WikidataQueryRun,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# RAG EMBEDDINGS
# ============================================================

_embeddings = None
_rag_available = False

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings

    _embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    _rag_available = True

except Exception:
    _rag_available = False


# Context variable for current thread_id
current_thread_id_ctx: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("thread_id", default=None)
)


# ============================================================
# TOOLS
# ============================================================

search_tool = DuckDuckGoSearchRun(
    region="in",
    safesearch="Moderate"
)

wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper()
)

wikidata = WikidataQueryRun(
    api_wrapper=WikidataAPIWrapper()
)


# ------------------------------------------------------------
# Calculator
# ------------------------------------------------------------

@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str
) -> dict:
    """
    Perform basic arithmetic operations.

    Supported operations:
    add, sub, mul, div
    """

    try:

        if operation == "add":
            result = first_num + second_num

        elif operation == "sub":
            result = first_num - second_num

        elif operation == "mul":
            result = first_num * second_num

        elif operation == "div":

            if second_num == 0:
                return {
                    "error": "Division by zero is not allowed."
                }

            result = first_num / second_num

        else:
            return {
                "error": (
                    "Invalid operation. "
                    "Use add, sub, mul, or div."
                )
            }

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# ------------------------------------------------------------
# Stock Price
# ------------------------------------------------------------

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Get stock price data using Alpha Vantage.
    """

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")

    if not api_key:
        return {
            "error": (
                "Alpha Vantage API key is not configured. "
                "Add ALPHAVANTAGE_API_KEY to your .env file."
            )
        }

    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_INTRADAY"
        f"&symbol={symbol}"
        f"&interval=5min"
        f"&apikey={api_key}"
    )

    try:

        response = requests.get(
            url,
            timeout=10
        )

        response.raise_for_status()

        return response.json()

    except Exception as e:

        return {
            "error": str(e)
        }


# ============================================================
# RAG / PDF SECTION
# ============================================================

_rag_vector_stores: dict[str, FAISS] = {}

_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

_RAG_DEPS_MSG = (
    "PDF Q&A is unavailable because the required dependencies "
    "are missing. Run:\n"
    "pip install -U huggingface_hub transformers sentence-transformers"
)


# ------------------------------------------------------------
# Load PDF
# ------------------------------------------------------------

def _load_pdf_documents(pdf_bytes: bytes) -> list[Document]:
    """
    Load PDF from bytes and convert pages into LangChain Documents.
    """

    reader = PdfReader(
        io.BytesIO(pdf_bytes)
    )

    docs = []

    for i, page in enumerate(reader.pages):

        text = page.extract_text()

        if text and text.strip():

            docs.append(
                Document(
                    page_content=text.strip(),
                    metadata={
                        "page": i + 1,
                        "source": "uploaded_pdf",
                    },
                )
            )

    return docs


# ------------------------------------------------------------
# Add PDF to RAG
# ------------------------------------------------------------

def add_pdf_for_thread(
    thread_id: str,
    pdf_bytes: bytes
) -> str:

    if not _rag_available or _embeddings is None:
        return _RAG_DEPS_MSG

    docs = _load_pdf_documents(pdf_bytes)

    if not docs:
        return "No text could be extracted from the PDF."

    chunks = _text_splitter.split_documents(
        docs
    )

    # Replace existing vector store
    if thread_id in _rag_vector_stores:
        del _rag_vector_stores[thread_id]

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=_embeddings,
    )

    _rag_vector_stores[thread_id] = vector_store

    return (
        f"PDF ingested successfully. "
        f"{len(chunks)} chunks from "
        f"{len(docs)} page(s). "
        "You can now ask questions about this document."
    )


# ------------------------------------------------------------
# Query Uploaded PDF
# ------------------------------------------------------------

@tool
def query_uploaded_document(
    question: str
) -> str:
    """
    Search the PDF uploaded by the user in the current chat.
    """

    if not _rag_available:
        return _RAG_DEPS_MSG

    thread_id = current_thread_id_ctx.get()

    if (
        not thread_id
        or thread_id not in _rag_vector_stores
    ):
        return (
            "No PDF has been uploaded in this chat. "
            "Ask the user to upload a PDF first."
        )

    vector_store = _rag_vector_stores[thread_id]

    results = vector_store.similarity_search(
        question,
        k=4
    )

    if not results:
        return (
            "No relevant passages were found "
            "in the uploaded document."
        )

    excerpts = "\n\n---\n\n".join(
        doc.page_content
        for doc in results
    )

    return (
        "Relevant excerpts from the uploaded PDF:\n\n"
        + excerpts
    )


# ============================================================
# TOOL LIST
# ============================================================

tools = [
    search_tool,
    calculator,
    get_stock_price,
    query_uploaded_document,
    wikipedia,
    wikidata,
]


# ============================================================
# GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.6,
)

llm = llm.bind_tools(tools)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Nexxa, a helpful and intelligent AI assistant
created by Deepanshu Bairagi.

Answer users' questions interactively.

Follow these instructions:

1. First determine whether a tool is needed.
   If a tool is clearly not required, answer directly.

2. If one tool fails, try another appropriate tool.
   For example, if DuckDuckGo search does not provide
   useful results, try Wikipedia or another suitable tool.

3. Follow BODMAS when solving mathematical calculations.

4. You can use multiple tools when necessary.

5. If the user has uploaded a PDF, use relevant information
   from that PDF when answering their questions.

6. For technical or study-related questions:
   answer the question first and then ask two related
   follow-up questions.

7. Answer in clear, organized points whenever appropriate.

8. Do not claim to have information that you do not have.

9. Be helpful, accurate, and concise.
"""


# ============================================================
# STATE
# ============================================================

class ChatState(TypedDict):

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# ============================================================
# CHAT NODE
# ============================================================

def chat_node(state: ChatState):

    """
    LLM node that can answer directly
    or request tool calls.
    """

    messages = state["messages"]

    # Add system prompt if it is not already present
    if (
        not messages
        or not isinstance(
            messages[0],
            SystemMessage
        )
    ):

        messages_with_system = [
            SystemMessage(
                content=SYSTEM_PROMPT
            )
        ] + messages

    else:

        messages_with_system = messages

    try:

        response = llm.invoke(
            messages_with_system
        )

        return {
            "messages": [response]
        }

    except Exception as e:

        err_msg = str(e)

        if "failed_generation" in err_msg:

            err_msg = (
                err_msg
                .split("failed_generation")[0]
                .strip()
            )

        return {
            "messages": [
                AIMessage(
                    content=(
                        "I ran into an issue while "
                        "handling that request: "
                        f"{err_msg}"
                    )
                )
            ]
        }


# ============================================================
# TOOL NODE
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# SQLITE CHECKPOINT
# ============================================================

conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(
    conn=conn
)


# ============================================================
# LANGGRAPH
# ============================================================

graph = StateGraph(
    ChatState
)

graph.add_node(
    "chat_node",
    chat_node
)

graph.add_node(
    "tools",
    tool_node
)


# START → CHAT

graph.add_edge(
    START,
    "chat_node"
)


# CHAT → TOOLS or END

graph.add_conditional_edges(
    "chat_node",
    tools_condition
)


# TOOLS → CHAT

graph.add_edge(
    "tools",
    "chat_node"
)


# Compile

chatbot = graph.compile(
    checkpointer=checkpointer
)


# ============================================================
# RETRIEVE ALL THREADS
# ============================================================

def retrieve_all_threads():

    all_threads = set()

    for checkpoint in checkpointer.list(None):

        thread_id = (
            checkpoint.config
            .get("configurable", {})
            .get("thread_id")
        )

        if thread_id:
            all_threads.add(thread_id)

    return list(all_threads)
