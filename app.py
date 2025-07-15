import io
import tempfile
import uuid
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from typing import List, Sequence
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.messages import ChatMessage, TextMessage
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from openai import AzureOpenAI
import os
import streamlit as st


from dotenv import load_dotenv

load_dotenv()

endpoint = os.environ["AZURE_OPENAI_ENDPOINT"] # Sample : https://<account_name>.services.ai.azure.com/api/projects/<project_name>
api_key= os.environ["AZURE_OPENAI_KEY"]
model_deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT"] 
WHISPER_DEPLOYMENT_NAME = "whisper"

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2024-06-01"  # Adjust API version as needed
)
# Define a model client. You can use other model client that implements
# the `ChatCompletionClient` interface.
model_client = AzureOpenAIChatCompletionClient(
    model=model_deployment_name,
    api_key=api_key,
    azure_endpoint=endpoint,
    deployment_name=model_deployment_name,
    api_version="2024-10-21",  # Specify the API version if needed.
    seed=42,  # Optional: Set a seed for reproducibility.
    temperature=0.0,  # Optional: Set the temperature for the model (0.0 to 1.0, where 0.0 is deterministic and 1.0 is more random).
)

def transcribe_audio(audio_data) -> str:
    """Transcribe audio using Azure OpenAI Whisper."""
    try:
        # Convert audio data to the format expected by Whisper
        audio_file = io.BytesIO(audio_data.getvalue())
        audio_file.name = "audio.wav"
        
        transcript = client.audio.transcriptions.create(
            model=WHISPER_DEPLOYMENT_NAME,
            file=audio_file
        )
        return transcript.text
    except Exception as e:
        st.error(f"❌ Audio transcription failed: {e}")
        return ""

def generate_audio_response_gpt_1(text, selected_voice):
    """Generate audio response using gTTS."""
    # tts = gTTS(text=text, lang="en")
    url = os.getenv("AZURE_OPENAI_ENDPOINT") + "/openai/deployments/gpt-4o-mini-tts/audio/speech?api-version=2025-03-01-preview"  
  
    headers = {  
        "Content-Type": "application/json",  
        "Authorization": f"Bearer {os.environ['AZURE_OPENAI_KEY']}"  
    }  

    prompt = f"""can you make this content as short and sweet rather than reading the text and make it personally to user to listen to it.
    Keep in conversation and with out confirming to user about the story telling.
    Also make the content few sentences long and make it more practical to user.
    {text}"""

    audioclient = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2025-03-01-preview"
    )

    # speech_file_path = Path(__file__).parent / "speech.mp3"
    temp_file = os.path.join(tempfile.gettempdir(), f"response_{uuid.uuid4()}.mp3")

    with audioclient.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=selected_voice.lower(), #"coral",
        input=text,
        instructions="Speak in a cheerful and positive tone. Can you make this content as story telling rather than reading the text and make it personally to user to listen to it.",
    ) as response:
        response
        response.stream_to_file(temp_file)

    return temp_file

async def sa_assist():

        # Configure Streamlit page with Material Design 3 theme
    st.set_page_config(
        page_title="Solution Architect Assist",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Create a system for Solution Architect to interact and get information
    planning_agent = AssistantAgent(
        "PlanningAgent",
        description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
        model_client=model_client,
        system_message="""
        You are a planning agent.
        Your job is to break down complex tasks into smaller, manageable subtasks.
        Your team members are:
            sa_business_agent: Provide business requirements and insights.
            sa_architect_agent: Create architectural designs and oversee implementation. Also create architectural diagrams.
            sa_analyst_agent: Provides SWOT, TOWS Matrix, PESTLE, Porter’s Five Forces,SOAR Analysis, VRIO Framework , SCORE Analysis, NOISE Analysis.
            mcp_fetch_agent: Fetch relevant Microsoft Learn content based on the architecture and provide relevant information.

        You only plan and delegate tasks - you do not execute them yourself.

        When assigning tasks, use this format:
        1. <agent> : <task>

        After all tasks are complete, summarize the findings and end with "TERMINATE".
        """,
    )

    sa_business_agent = AssistantAgent(
        "SABusinessAgent",  
        description="An agent for business-related tasks, this agent should be the second to engage when given a new task.",
        model_client=model_client,
        system_message="""
        You are a business agent.
        Your job is to collect business requirements and provide insights.
        """,
    )

    sa_architect_agent = AssistantAgent(
        "SAArchitectAgent", 
        description="An agent for architectural tasks, this agent should be the third to engage when given a new task.",
        model_client=model_client,
        system_message="""
        You are an architectural agent. Create the architecture based on Azure PaaS services with AI first design.
        Provide Secure by Design principles and best practices.
        Provide architectural diagrams and design documents.
        Your job is to design and oversee the implementation of technical solutions.
        """,
    )

    sa_analyst_agent = AssistantAgent(
        "SAAnalystAgent",
        description="""An agent for analytical tasks, Do a SWOT, TOWS Matrix, PESTLE, Porter’s Five Forces,SOAR Analysis
        VRIO Framework , SCORE Analysis, NOISE Analysis
        Other analysis.""",
        model_client=model_client,
        system_message="""
        You are an analyst agent.
        Your job is to analyze data and provide insights.
        """,
    )

    # Get the fetch tool from mcp-server-fetch.
    fetch_mcp_server = StdioServerParams(command="uvx", args=["https://learn.microsoft.com/api/mcp"])
    sa_mslearn_doc = McpWorkbench(fetch_mcp_server)
    # Create an agent that can use the fetch tool.
    mcp_fetch_agent = AssistantAgent(
        name="mcp_fetcher", model_client=model_client, workbench=sa_mslearn_doc, reflect_on_tool_use=True,
        system_message="""
        You are a Microsoft Learn content agent. Use the current content based on the architecture and provide relevant information.
        """
    )

    # create a selectorgroup chat for courses here
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=25)
    termination = text_mention_termination | max_messages_termination

    selector_prompt = """Select an agent to perform task.

    {roles}

    Current conversation context:
    {history}

    Read the above conversation, then select an agent from {participants} to perform the next task.
    Make sure the planner agent has assigned tasks before other agents start working.
    Only select one agent.
    """
    team = SelectorGroupChat(
        [planning_agent, sa_business_agent, sa_architect_agent, sa_analyst_agent, mcp_fetch_agent],
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=True,  # Allow an agent to speak multiple turns in a row.
    )
    # Create the final reviewer agent
    final_reviewer = AssistantAgent(
        "final_reviewer",
        model_client=model_client,
        system_message="Consolidate the course information and summarize.",
    )

    # Build the workflow graph
    learner_agent_graph = DiGraphBuilder()
    learner_agent_graph.add_node(planning_agent).add_node(sa_business_agent).add_node(sa_architect_agent).add_node(sa_analyst_agent).add_node(final_reviewer)
    # Fan-out from writer to editor1 and editor2
    learner_agent_graph.add_edge(planning_agent, sa_business_agent)
    learner_agent_graph.add_edge(planning_agent, sa_architect_agent)
    learner_agent_graph.add_edge(planning_agent, sa_analyst_agent)
    # Fan-in both editors into final reviewer
    learner_agent_graph.add_edge(sa_business_agent, final_reviewer)
    learner_agent_graph.add_edge(sa_architect_agent, final_reviewer)
    learner_agent_graph.add_edge(sa_analyst_agent, final_reviewer)

    task = "Create a Email agent to read my email and prioritize messages?"
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize session state to track processed audio
    if "processed_audio" not in st.session_state:
        st.session_state.processed_audio = set()
    
    # Initialize session state to track if we're currently processing
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # Create a scrollable container for chat history
    chat_container = st.container(height=400)
    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # If this is an assistant message and has an audio file, display it
                if message["role"] == "assistant" and "audio_file" in message:
                    st.audio(message["audio_file"], format="audio/mp3")

    # Build and validate the graph
    graph = learner_agent_graph.build()

    # Create the flow
    flow = GraphFlow(
        participants=learner_agent_graph.get_participants(),
        graph=graph,
    )


    # Input methods section
    st.markdown("---")
    st.markdown("### 💬 Ask your question")
    st.markdown("Choose your preferred input method:")
    
    # Create columns for audio and text input
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown("**🎤 Voice Input**")
        audio_data = st.audio_input("Record your question")
    
    with col2:
        st.markdown("**⌨️ Text Input**")
        prompt = st.chat_input("Type your message here or use voice recording above...")

    # Process audio input
    if audio_data and not st.session_state.processing:
        # Create a unique identifier for this audio input
        audio_id = str(hash(audio_data.getvalue()))
        
        # Only process if we haven't processed this audio before
        if audio_id not in st.session_state.processed_audio:
            st.session_state.processing = True
            st.session_state.processed_audio.add(audio_id)
            
            with st.spinner("🎤 Transcribing audio..."):
                transcriptiontext = transcribe_audio(audio_data)
            
            if transcriptiontext:
                # Add the transcribed text to chat history
                st.session_state.messages.append({"role": "user", "content": f"🎤 {transcriptiontext}"})
                
                with st.spinner("🤖 Processing your request..."):
                    # Use the same team workflow as text input for consistency
                    result = await Console(team.run_stream(task=transcriptiontext))
                    # result = await Console(flow.run_stream(task=transcriptiontext))
                    last_message = None
                    for message in result.messages:
                        if hasattr(message, 'content') and message.content:
                            last_message = message.content
                
                if last_message:
                    with st.spinner("🔊 Generating audio response..."):
                        audio_file = generate_audio_response_gpt_1(last_message, "coral")
                    st.session_state.messages.append({"role": "assistant", "content": last_message, "audio_file": audio_file})
            
            st.session_state.processing = False
            st.rerun()

    # Process text input
    if prompt and not st.session_state.processing:
        st.session_state.processing = True
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("🤖 Processing your request..."):
            # Generate assistant response using the team workflow
            result = await Console(team.run_stream(task=prompt))
            
            # Parse the output to get the last summarized message
            last_message = None
            for message in result.messages:
                if hasattr(message, 'content') and message.content:
                    last_message = message.content

        # Add assistant response to chat history and generate audio
        if last_message:
            with st.spinner("🔊 Generating audio response..."):
                audio_file = generate_audio_response_gpt_1(last_message, "coral")
            st.session_state.messages.append({"role": "assistant", "content": last_message, "audio_file": audio_file})
        
        st.session_state.processing = False
        st.rerun()

    # Add a clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.processed_audio = set()  # Clear processed audio tracking
        st.session_state.processing = False  # Reset processing state
        st.rerun()

if __name__ == "__main__":
    import asyncio
    # NOTE: if running this inside a Python script you'll need to use asyncio.run(main_orc()).
    asyncio.run(sa_assist())
