import streamlit as st

from langchain.document_loaders import PDFPlumberLoader
from langchain.document_transformers import Html2TextTransformer

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder
from langsmith import Client

from secret import openai_api_key

client = Client()

st.set_page_config(
    page_title="ChatLangChain",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="collapsed",
)

"# Chat🦜🔗"

llm = ChatOpenAI(
    temperature=0, openai_api_key=openai_api_key, streaming=True, model="gpt-4"
)
base_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)


@st.cache_resource(ttl="1h")
def configure_retriever():
    # Load and convert PDF to text
    loader = PDFPlumberLoader("KPM Offsite 2021 PDF .pdf")
    docs = loader.load()
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(docs)

    # Create prompt for HyDE
    prompt_template = """
Please answer the user's question about Kroger's KPM Offsite advertising.
Question: {question}
Answer:
"""
    prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    # Create HyDE embeddings
    embeddings = HypotheticalDocumentEmbedder(
        llm_chain=llm_chain,
        base_embeddings=base_embeddings,
    )
    # Create vectorstore
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


tool = create_retriever_tool(
    configure_retriever(),
    "search_KPM-offsite_docs",
    "Searches and returns documents regarding Kroger KPM Offsite. Kroger KPM Offsite advertising is a way to advertise CPG brands to Kroger's customers utilizing first-party data. Always check your responses to Kroger offsite advertising questions by using this tool.",
)
tools = [tool]

message = SystemMessage(
    content=(
        "You are a helpful chatbot who is tasked with answering questions about Kroger Advertising KPM. "
        "Unless otherwise explicitly stated, it is probably fair to assume that questions are about Kroger Advertising KPM. "
        "If there is any ambiguity, you probably assume they are about that."
    )
)
prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
)
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
)
memory = AgentTokenBufferMemory(llm=llm)
starter_message = "Ask me anything about Kroger Advertising!"
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [AIMessage(content=starter_message)]


def send_feedback(run_id, score):
    client.create_feedback(run_id, "user_score", score=score)


for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    memory.chat_memory.add_message(msg)


if prompt := st.chat_input(placeholder=starter_message):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor(
            {"input": prompt, "history": st.session_state.messages},
            callbacks=[st_callback],
            include_run_info=True,
        )
        st.session_state.messages.append(AIMessage(content=response["output"]))
        st.write(response["output"])
        memory.save_context({"input": prompt}, response)
        st.session_state["messages"] = memory.buffer
        run_id = response["__run"].run_id

        col_blank, col_text, col1, col2 = st.columns([10, 2, 1, 1])
        with col_text:
            st.text("Feedback:")

        with col1:
            st.button("👍", on_click=send_feedback, args=(run_id, 1))

        with col2:
            st.button("👎", on_click=send_feedback, args=(run_id, 0))
