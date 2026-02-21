import streamlit as st
from backend import chatbot, rerieve_all_threads, add_pdf_for_thread, current_thread_id_ctx
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
import uuid

# ***************  UTILITY FUNCTIONS *****************
def generate_thread_id():
    tid = uuid.uuid4()
    return tid

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"])
    st.session_state["history"] = []

def add_thread(thread_id):
    if st.session_state["chat_threads"] is None:
        st.session_state["chat_threads"] = []
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config = {"configurable":{"thread_id": thread_id}})
    if state.values:
        return state.values.get("messages", [])
    return []
    
        
        
st.set_page_config(
    page_title="Srii",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Srii")


# *************** SESSION SETUP ******************
if "history" not in st.session_state:
    st.session_state["history"] = [{"role": "ai", "content": "Hello! How can I assist you today?"}]

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = rerieve_all_threads()

add_thread(st.session_state["thread_id"])

CONFIG = {"configurable":{"thread_id": st.session_state["thread_id"]}}

# *************** SIDEBAR UI *****************
st.sidebar.title("Srii's Tools 🤖")

# PDF upload for RAG
with st.sidebar.expander("📄 Upload PDF (ask questions about it)"):
    uploaded_pdf = st.file_uploader("Choose a PDF", type=["pdf"], key="pdf_upload")
    if uploaded_pdf is not None and st.button("Process PDF", key="process_pdf_btn"):
        pdf_bytes = uploaded_pdf.read()
        with st.spinner("Processing PDF..."):
            result = add_pdf_for_thread(str(st.session_state["thread_id"]), pdf_bytes)
        st.success(result)
        st.caption("You can now ask questions about this PDF in the chat.")

with st.sidebar:
    if st.button("🧹 Clear Chat"):
        st.session_state.history = []
        
    with st.expander("About Srii"):
        st.markdown("""
            - Siri ❌ Srii ✅
            - Made by Ajay
            - Using LangGraph 
            """)
    st.set_page_config(
    page_title="Srii",
    page_icon="🤖",
    layout="centered"
    )
if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Chats")

for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        temp_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                temp_messages.append({"role": "user", "content": message.content})
            else:
                temp_messages.append({"role": "ai", "content": message.content})
        st.session_state["history"] = temp_messages



### this loop is displaying the conversation history. We re-render the entire history on every new message for simplicity, but in a production app you'd want to optimize this.
for message in st.session_state["history"]:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


user_input = st.chat_input("Type your message here...")

if user_input:
    st.toast("Message sent 🚀")

    st.session_state["history"].append({"role": "user", "content": user_input})
    with st.chat_message("human"):
        st.markdown(user_input)
    

    ### Assistant streaming block with status
    with st.chat_message("assistant"):
        with st.status("Srii is working...", expanded=True) as status:
            status_placeholder = st.empty()
            message_placeholder = st.empty()
            full_content = ""

            # Set thread_id in context so RAG tool can use the correct vector store
            token = current_thread_id_ctx.set(str(st.session_state["thread_id"]))
            try:
                for message_chunk, metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]}, config=CONFIG,
                    stream_mode="messages"
                ):
                    node = metadata.get("langgraph_node", "")
                    if node == "chat_node":
                        status_placeholder.caption("💭 Thinking...")
                    elif node == "tools":
                        if isinstance(message_chunk, ToolMessage):
                            tool_name = getattr(message_chunk, "name", "tool") or "tool"
                            status_placeholder.caption(f"🔧 Using: **{tool_name}**")
                        else:
                            status_placeholder.caption("🔧 Using tools...")

                    if isinstance(message_chunk, AIMessage) and message_chunk.content:
                        full_content += message_chunk.content
                        message_placeholder.markdown(full_content + "▌")
            finally:
                current_thread_id_ctx.reset(token)
            # Update status so "Srii is working..." is replaced and box collapses
            status.update(label="✅ Done", state="complete", expanded=True)
            message_placeholder.markdown(full_content)

        ai_message = full_content

    st.session_state["history"].append({"role": "assistant", "content": ai_message})

