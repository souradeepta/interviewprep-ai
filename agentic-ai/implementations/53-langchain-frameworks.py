"""
Auto-generated from 53-langchain-frameworks.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # 53 Langchain Frameworks
# ## Learning Objectives
# 1. Understand core concepts of 53 langchain frameworks
# 2. Implement with production libraries
# ======================================================================

# ======================================================================
# ## Level 1: Basic Implementation
# ======================================================================

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize LLM
llm = OpenAI(temperature=0.9)

# Create a simple prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Tell me a fun fact about {topic}",
)

# Create a chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
result = chain.run(topic="machine learning")
print(f"Result: {result}")

# Verify components
print(f"\nLLM: {llm.__class__.__name__}")
print(f"Prompt variables: {prompt.input_variables}")
print(f"Chain type: {chain.__class__.__name__}")



# ======================================================================
# ## Level 2: Advanced Implementation
# ======================================================================

from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

# Initialize components
llm = OpenAI(temperature=0.7)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
memory = ConversationBufferMemory(memory_key="chat_history")

# Create agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Run agent
response = agent.run(input="What is 2+2? Then search for recent AI news")

print(f"\nAgent Response: {response}")
print(f"Memory: {memory.buffer}")



# ======================================================================
# ## Real-World Example 1: Production Pattern
# ======================================================================

# Real-World: Build a Multi-Step Research Agent
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Tools for research
llm = OpenAI(temperature=0.7)
tools = load_tools(["serpapi", "summarization"], llm=llm)

# Agent for research
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.REACT,
    verbose=False
)

# Research prompt
research_prompt = PromptTemplate(
    template="Research and summarize: {query}",
    input_variables=["query"]
)

# Multi-step workflow
queries = [
    "Latest breakthroughs in transformer models",
    "Top papers on agents",
    "Comparison of frameworks"
]

results = []
for query in queries:
    result = agent.run(research_prompt.format(query=query))
    results.append(result)

print(f"Researched {len(results)} topics")
print(f"First result preview: {results[0][:100]}...")



# ======================================================================
# ## Real-World Example 2: Advanced Usage
# ======================================================================

# Real-World: Document Question-Answering
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Load documents
loader = TextLoader("document.txt")
documents = loader.load()

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# Create QA chain
llm = OpenAI(temperature=0)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Ask questions
questions = [
    "What is the main topic?",
    "What are key points?",
]

for q in questions:
    answer = qa.run(q)
    print(f"Q: {q}\nA: {answer}\n")



# ======================================================================
# ## Real-World Example 3: Optimization
# ======================================================================

# Real-World: Custom Tools and Tool Composition
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.llms import OpenAI

# Define custom tools
def calculate_product(a: str) -> str:
    '''Multiply two numbers'''
    nums = a.split('*')
    return str(float(nums[0]) * float(nums[1]))

def word_count(text: str) -> str:
    '''Count words in text'''
    return str(len(text.split()))

# Create Tool objects
tools = [
    Tool(name="Calculator", func=calculate_product, description="Multiply numbers"),
    Tool(name="WordCount", func=word_count, description="Count words"),
]

# Create agent with custom tools
llm = OpenAI(temperature=0.7)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Test agent
result = agent.run("What is 5*3? Count words: 'Hello world'")
print(f"Result: {result}")




# ======================================================================
# ## Key Takeaways
# **When to use 53 langchain frameworks:**
# - Understand when this concept applies
# - Consider tradeoffs and constraints
# ======================================================================
