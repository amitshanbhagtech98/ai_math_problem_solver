"""
Math Problem Solver and Data Search Assistant
------------------------------------------
This application is a sophisticated math problem solver and information search tool built using:
- Streamlit for the web interface
- LangChain for orchestrating the language model and tools
- Groq's Gemma-2 model as the base language model
- Various tools including Wikipedia search and calculator functionalities

The app can:
1. Solve mathematical problems with detailed explanations
2. Search Wikipedia for relevant information
3. Provide logical reasoning for complex problems
"""

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq  # Groq's language model integration
from langchain.chains import LLMMathChain, LLMChain  # Chain components for logic flow
from langchain.prompts import PromptTemplate  # Template for structured prompts
from langchain_community.utilities import WikipediaAPIWrapper  # Wikipedia search capability
from langchain.agents.agent_types import AgentType  # Agent type definitions
from langchain.agents import Tool, initialize_agent  # Agent and tool components
from langchain.callbacks import StreamlitCallbackHandler  # Streamlit integration for callbacks

load_dotenv()

# Set up LangSmith tracking (for experiment management and tracing)
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_TRACING'] = "true"
os.environ['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT')

# Initialize the Streamlit application
st.set_page_config(
    page_title="Text To Math Problem Solver And Data Search Assistant", 
    page_icon="🧮"  # Calculator emoji as page icon
)
st.title("Text To Math Problem Solver Using Gemma 2")

# API Key input in sidebar for security
groq_api_key = st.sidebar.text_input(label="Groq API Key", type="password")

# Ensure API key is provided before proceeding
if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()

# Initialize the language model with Groq's Gemma-2 model
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)


# Initialize tool components
# 1. Wikipedia Search Tool
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching Wikipedia to find various information on the topics mentioned"
)

# 2. Mathematical Computation Tool

math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tools for answering math related questions. Only input mathematical expression need to bed provided"
)

# Define the prompt template for mathematical reasoning
prompt = """
You are an agent tasked with solving user's mathematical questions. 
Please logically arrive at the solution and provide a detailed explanation
in a point-wise format for the question below:

Question: {question}
Answer:
"""

# Create a structured prompt template for consistency
prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

# Create a chain combining the LLM with the prompt template
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)

# Initialize the main agent with all available tools
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],  # Combine all tools
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Use zero-shot learning approach
    verbose=False,  # Disable verbose output
    handle_parsing_errors=True  # Gracefully handle parsing errors
)

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant",
         "content": "Hi, I'm a Math chatbot who can help you solve mathematical problems!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# User Interface Section
# Text input area with a default example question
question = st.text_area(
    "Enter your question:", 
    "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. " +
    "Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries " +
    "contains 25 berries. How many total pieces of fruit do I have at the end?"
)

# Process the question when the button is clicked
if st.button("Find My Answer"):
    if question:
        # Show a loading spinner while generating the response
        with st.spinner("Generating response..."):
            # Add user's question to chat history
            st.session_state.messages.append(
                {"role": "user", "content": question})
            st.chat_message("user").write(question)

            # Create a Streamlit callback handler for real-time updates
            st_cb = StreamlitCallbackHandler(
                st.container(), 
                expand_new_thoughts=False
            )
            
            # Generate response using the assistant agent
            response = assistant_agent.run(
                st.session_state.messages, 
                callbacks=[st_cb]
            )
            
            # Add assistant's response to chat history
            st.session_state.messages.append(
                {'role': 'assistant', "content": response})
            
            # Display the response
            st.write('### Response:')
            st.success(response)
    else:
        # Show warning if no question is entered
        st.warning("Please enter a question")
