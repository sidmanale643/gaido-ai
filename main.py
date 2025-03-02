import streamlit as st
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core import VectorStoreIndex

from llama_index.core.query_engine import RetrieverQueryEngine
import os
from llama_index.core import Settings , Document 
from dotenv import load_dotenv
from llama_index.retrievers.bm25 import BM25Retriever
import nest_asyncio
import streamlit as st
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import nest_asyncio
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.llms.groq import Groq
from llama_index.core.node_parser import SentenceSplitter
import pandas as pd
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
llama_parse_api_key = os.getenv("LLAMA_PARSE_API_KEY")

#llm = Groq(model = "llama-3.1-8b-instant", api_key = "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" )
#embed_model = HuggingFaceEmbedding(model_name = "nomic-ai/nomic-embed-text-v1" ,trust_remote_code=True)

llm = Ollama(model="llama3.2:3b", request_timeout=120.0)

embed_model = OllamaEmbedding(
    model_name="jina/jina-embeddings-v2-base-en",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

Settings.llm = llm
Settings.embed_model = embed_model

# prompt = """Key Fields to Extract:
# Policyholder Info: Name, Address, Contact
# Policy Details: Policy Number, Customer ID, Start & Expiry Dates, Plan Type, Renewal Date
# Coverage: Base Sum Insured, Safeguard, Booster Benefit, Total Sum Insured
# Premium: Net Premium, Taxes, Gross Premium (in numbers & words)
# Nominee & Intermediary: Name, Relationship, Intermediary Contact
# Claims & Grievance: Settlement Time, Submission Process, Customer Support Links
# Exclusions & Waiting Periods: Pre-existing Conditions, Specific Waiting Times, Permanent Exclusions
# Benefits & Riders: Cashless Claims, Room Rent, Air Ambulance, Health Checkups, No-Claim Bonus
# """


# documents_with_instruction = LlamaParse(
#     result_type="markdown",
#     api_key = llama_parse_api_key,
#     content_guideline_instruction = prompt
#     ).load_data(r"C:\Users\SiD\vs\GaidoAI\Sample HI Policy.pdf")

documents = SimpleDirectoryReader(input_files=[r'GaidoAI\insurance_documents.md']).load_data()

index = VectorStoreIndex.from_documents(
    documents=documents,
    show_progress=True,
    transformations=[SentenceSplitter(chunk_size= 512, chunk_overlap=128)],
    num_workers = 1,
)

idx_eng = index.as_query_engine(similarity_top_k=10),
bm_ret = BM25Retriever.from_defaults(index = index , similarity_top_k = 10)


if "memory" in st.session_state:
    memory = st.session_state["memory"]
else:
    memory = ChatSummaryMemoryBuffer.from_defaults(token_limit=3900)
    st.session_state["memory"] = memory

if "chat_history" in st.session_state:
    chat_history = st.session_state["chat_history"]
else:
    chat_history = []
    st.session_state["chat_history"] = chat_history

Settings.llm = llm
Settings.embed_model = embed_model

nest_asyncio.apply()

retriever = QueryFusionRetriever(
    [
        index.as_retriever(similarity_top_k=10),
        bm_ret,
    ],
    num_queries=1,
    use_async=True,
)

response_synthesizer = get_response_synthesizer(structured_answer_filtering=False)
engine = RetrieverQueryEngine(retriever , response_synthesizer= response_synthesizer)

st.set_page_config(layout="wide", page_title="Welcome Chatbot")
st.title("Welcome!", anchor=False)
st.write("Hi! I am your AI assistant. How can I help you?")

if "memory" not in st.session_state:
    st.session_state["memory"] = ChatSummaryMemoryBuffer.from_defaults(token_limit=3900)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

nest_asyncio.apply()


query = st.chat_input("Type your message here...")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": "Hi, I am your AI assistant. How can I help you?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    response = engine.query(query).response

    st.session_state.messages.append({"role": "ai", "content": response})
    with st.chat_message("ai"):
        st.write(response)