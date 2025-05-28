import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from utils.tools import get_custom_tools

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    model_kwargs={"seed": 42},
)

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)

# Load and process documents
def load_documents(directory="./documents"):
    loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks

# Create vector store
def create_vectorstore(documents):
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore

# Set up RAG pipeline
def setup_rag():
    # Load documents and create vector store
    documents = load_documents()
    vectorstore = create_vectorstore(documents)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Create retriever tool
    retriever_tool = create_retriever_tool(
        retriever,
        name="document_search",
        description="Searches for information in the document repository. Use this when you need to answer questions about specific documents or topics."
    )
    
    # Get custom tools
    custom_tools = get_custom_tools()
    
    # Combine all tools
    tools = [retriever_tool] + custom_tools
    
    # Create agent with MCP (Model Context Protocol)
    prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that helps users find information using the tools provided. "
              "Always use the appropriate tool when needed. If you don't know the answer, say so."),
    ("human", "{input}"),
    ("assistant", "{agent_scratchpad}")
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

# Simple RAG chain without agent
def setup_simple_rag():
    documents = load_documents()
    vectorstore = create_vectorstore(documents)
    retriever = vectorstore.as_retriever()
    
    template = """Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    return rag_chain

def main():
    # Set up the RAG agent with tools
    agent_executor = setup_rag()
    
    # Interactive loop
    print("Welcome to the RAG Assistant! Type 'exit' to quit.")
    while True:
        user_input = input("\nQuestion: ")
        if user_input.lower() == "exit":
            break
        
        response = agent_executor.invoke({"input": user_input})
        print(f"\nAssistant: {response['output']}")

if __name__ == "__main__":
    main()