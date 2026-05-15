import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
import chromadb

st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

st.sidebar.title("📚 Setup")
llm_choice = st.sidebar.selectbox("Choose LLM Backend", ["Ollama (Local)", "Claude (API)"])

llm = None
if llm_choice == "Ollama (Local)":
    ollama_model = st.sidebar.text_input("Ollama Model Name", value="llama3")
    base_url = st.sidebar.text_input("Ollama Base URL", value="http://localhost:11434")
    try:
        llm = ChatOllama(model=ollama_model, base_url=base_url)
    except Exception as e:
        st.sidebar.error(f"Error loading Ollama: {e}")

elif llm_choice == "Claude (API)":
    anthropic_api_key = st.sidebar.text_input("Anthropic API Key", type="password")
    if anthropic_api_key:
        llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            anthropic_api_key=anthropic_api_key,
            temperature=0
        )
    else:
        st.sidebar.warning("Please enter your Anthropic API key.")

st.sidebar.markdown("---")
st.sidebar.subheader("Document Upload")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if st.sidebar.button("Process Documents"):
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF file.")
    else:
        with st.spinner("Processing documents..."):
            documents = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                loader = PyPDFLoader(tmp_file_path)
                documents.extend(loader.load())
                os.remove(tmp_file_path)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            chroma_client = chromadb.Client()
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                client=chroma_client
            )
            
            st.session_state.vector_store = vector_store
            st.sidebar.success(f"Processed {len(uploaded_files)} files into {len(chunks)} chunks!")

st.title("📚 Chat with your Documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("Sources"):
                for i, doc in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}:** {doc.page_content}")
                    st.markdown(f"*Metadata: {doc.metadata}*")

if prompt := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not llm:
            st.error("Please configure the LLM backend in the sidebar.")
        elif st.session_state.vector_store is None:
            st.error("Please upload and process documents first.")
        else:
            with st.spinner("Thinking..."):
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                
                system_prompt = (
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, say that you "
                    "don't know. Use three sentences maximum and keep the "
                    "answer concise.\n\n"
                    "{context}"
                )
                
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])
                
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
                
                rag_chain_from_docs = (
                    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )
                
                rag_chain_with_source = RunnablePassthrough.assign(
                    context=(lambda x: retriever.invoke(x["input"]))
                ).assign(
                    answer=rag_chain_from_docs
                )
                
                response = rag_chain_with_source.invoke({"input": prompt})
                answer = response["answer"]
                source_documents = response["context"]
                
                st.markdown(answer)
                
                with st.expander("Sources"):
                    for i, doc in enumerate(source_documents):
                        st.markdown(f"**Source {i+1}:** {doc.page_content}")
                        st.markdown(f"*Metadata: {doc.metadata}*")
                        
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": source_documents
                })
