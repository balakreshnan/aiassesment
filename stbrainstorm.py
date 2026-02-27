import io
import tempfile
import uuid
import datetime
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
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from dotenv import load_dotenv

load_dotenv()

endpoint = os.environ["AZURE_OPENAI_ENDPOINT"] # Sample : https://<account_name>.services.ai.azure.com/api/projects/<project_name>
api_key= os.environ["AZURE_OPENAI_KEY"]
model_deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT"] 
WHISPER_DEPLOYMENT_NAME = "whisper"

# Initialize Azure OpenAI client
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"   # audience / scope for Azure OpenAI
)

client = AzureOpenAI(
    azure_endpoint=endpoint,
    # api_key=api_key,
    azure_ad_token_provider=token_provider,  # Use Azure AD authentication
    api_version="2024-10-21",  # Adjust API version as needed
)

# Define a model client
model_client = AzureOpenAIChatCompletionClient(
    model=model_deployment_name,
    # api_key=api_key,
    azure_ad_token_provider=token_provider,
    azure_endpoint=endpoint,
    deployment_name=model_deployment_name,
    api_version="2024-10-21",
    seed=42,
    temperature=0.7,  # Higher temperature for creative brainstorming
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

def generate_audio_response(text, selected_voice="coral"):
    """Generate audio response using Azure OpenAI TTS."""
    try:
        audioclient = AzureOpenAI(
            azure_endpoint=endpoint,
            # api_key=api_key,
            azure_ad_token_provider=token_provider,
            api_version="2025-03-01-preview"
        )

        temp_file = os.path.join(tempfile.gettempdir(), f"brainstorm_{uuid.uuid4()}.mp3")

        with audioclient.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=selected_voice.lower(),
            input=text,
            instructions="Speak in an engaging and enthusiastic tone suitable for brainstorming sessions.",
        ) as response:
            response.stream_to_file(temp_file)

        return temp_file
    except Exception as e:
        st.error(f"❌ Audio generation failed: {e}")
        return None

async def brainstorm_assistant():
    
    st.set_page_config(
        page_title="🧠 AI Brainstorming Studio",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        color: #4a90e2;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #4a90e2, #f39c12, #e74c3c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .brainstorm-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    .idea-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 0.5rem 0;
        color: white;
        border-left: 5px solid #ffffff;
    }
    .analysis-section {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 0.5rem 0;
        color: #2d3748;
    }
    .stExpander > div > div > div > div {
        background-color: #f8f9fa;
    }
    .voice-controls {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🧠 AI Brainstorming Studio</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.3rem; color: #666;">Transform your ideas into actionable business opportunities with AI-powered collaborative brainstorming</p>', unsafe_allow_html=True)
    
    # Sidebar with agent information
    with st.sidebar:
        st.markdown("## 🎯 Brainstorming Team")
        
        agent_info = {
            "🤔 Ideation Catalyst": {
                "role": "Creative idea generation and expansion",
                "expertise": "Innovation, creativity, lateral thinking",
                "tasks": "Generate ideas, ask provocative questions, explore possibilities"
            },
            "❓ Inquiry Specialist": {
                "role": "Strategic questioning and deep analysis",
                "expertise": "Critical thinking, analysis, research",
                "tasks": "Ask follow-up questions, probe deeper, uncover insights"
            },
            "💼 Business Analyst": {
                "role": "Market and financial analysis",
                "expertise": "Revenue models, market sizing, competitive analysis",
                "tasks": "Analyze market potential, revenue streams, business viability"
            },
            "🚀 Technology Advisor": {
                "role": "Technology trends and implementation",
                "expertise": "Emerging tech, digital transformation, innovation",
                "tasks": "Recommend technologies, assess technical feasibility"
            },
            "📊 Strategic Analyst": {
                "role": "SWOT, PESTEL, and strategic analysis",
                "expertise": "Strategic planning, risk assessment, market analysis",
                "tasks": "Conduct strategic analysis, identify opportunities and threats"
            },
            "📋 Resource Planner": {
                "role": "Resource allocation and planning",
                "expertise": "Project management, resource optimization, budgeting",
                "tasks": "Plan resources, estimate costs, create timelines"
            },
            "🎯 Success Metrics Expert": {
                "role": "KPI definition and success measurement",
                "expertise": "Performance metrics, analytics, monitoring",
                "tasks": "Define success criteria, create measurement frameworks"
            }
        }
        
        for agent_name, info in agent_info.items():
            with st.expander(agent_name):
                st.markdown(f"**Role:** {info['role']}")
                st.markdown(f"**Expertise:** {info['expertise']}")
                st.markdown(f"**Tasks:** {info['tasks']}")

    # Create specialized brainstorming agents
    ideation_agent = AssistantAgent(
        "IdeationCatalyst",
        description="Creative ideation and innovation catalyst",
        model_client=model_client,
        system_message="""
        You are an Ideation Catalyst, a creative powerhouse for brainstorming sessions.
        
        Your role is to:
        - Generate creative and innovative ideas
        - Expand on initial concepts with fresh perspectives
        - Ask thought-provoking questions to stimulate creativity
        - Encourage out-of-the-box thinking
        - Build upon ideas to create new possibilities
        
        Structure your responses as:
        ## 💡 Creative Insights
        ### Initial Ideas
        - [List 3-5 innovative ideas]
        ### Expansion Opportunities
        - [Ways to expand or modify ideas]
        ### Provocative Questions
        - [Questions to spark further creativity]
        
        Be enthusiastic, creative, and push boundaries while remaining practical.
        """,
    )

    inquiry_agent = AssistantAgent(
        "InquirySpecialist",
        description="Strategic questioning and deep analysis specialist",
        model_client=model_client,
        system_message="""
        You are an Inquiry Specialist, focused on asking the right questions to uncover insights.
        
        Your role is to:
        - Ask strategic follow-up questions
        - Probe deeper into assumptions and ideas
        - Uncover hidden opportunities and challenges
        - Challenge thinking to strengthen concepts
        - Guide discovery through targeted questioning
        
        Structure your responses as:
        ## ❓ Strategic Inquiry
        ### Key Questions to Explore
        - [5-7 strategic questions]
        ### Assumptions to Validate
        - [Critical assumptions that need testing]
        ### Areas for Deep Dive
        - [Topics requiring further investigation]
        
        Focus on questions that lead to actionable insights and better understanding.
        """,
    )

    business_analyst = AssistantAgent(
        "BusinessAnalyst",
        description="Market and financial analysis expert",
        model_client=model_client,
        system_message="""
        You are a Business Analyst specializing in market and financial analysis.
        
        Your role is to:
        - Analyze market potential and sizing
        - Evaluate revenue models and financial viability
        - Assess competitive landscape
        - Identify target customer segments
        - Evaluate business model feasibility
        
        Structure your responses as:
        ## 💼 Business Analysis
        ### Market Opportunity
        - Market size and growth potential
        - Target customer segments
        ### Revenue Model
        - Potential revenue streams
        - Pricing strategies
        ### Competitive Landscape
        - Key competitors and differentiation
        ### Financial Viability
        - Investment requirements and ROI projections
        
        Provide data-driven insights and realistic business assessments.
        """,
    )

    tech_advisor = AssistantAgent(
        "TechnologyAdvisor",
        description="Technology trends and implementation expert",
        model_client=model_client,
        system_message="""
        You are a Technology Advisor focused on emerging technologies and implementation.
        
        Your role is to:
        - Recommend relevant emerging technologies
        - Assess technical feasibility
        - Identify technology trends and opportunities
        - Suggest implementation approaches
        - Evaluate technical risks and mitigation strategies
        
        Structure your responses as:
        ## 🚀 Technology Recommendations
        ### Emerging Technologies
        - [Relevant cutting-edge technologies]
        ### Implementation Approach
        - [Technical architecture and approach]
        ### Technology Stack
        - [Recommended tools and platforms]
        ### Innovation Opportunities
        - [Ways to leverage technology for competitive advantage]
        
        Focus on practical, market-ready technology solutions.
        """,
    )

    strategic_analyst = AssistantAgent(
        "StrategicAnalyst",
        description="SWOT, PESTEL, and strategic analysis expert",
        model_client=model_client,
        system_message="""
        You are a Strategic Analyst specializing in comprehensive strategic analysis.
        
        Your role is to:
        - Conduct SWOT analysis (Strengths, Weaknesses, Opportunities, Threats)
        - Perform PESTEL analysis (Political, Economic, Social, Technological, Environmental, Legal)
        - Identify strategic opportunities and risks
        - Assess market positioning
        - Evaluate strategic alternatives
        
        Structure your responses as:
        ## 📊 Strategic Analysis
        ### SWOT Analysis
        - **Strengths**: [Internal advantages]
        - **Weaknesses**: [Internal challenges]
        - **Opportunities**: [External possibilities]
        - **Threats**: [External risks]
        ### PESTEL Analysis
        - **Political**: [Political factors]
        - **Economic**: [Economic conditions]
        - **Social**: [Social trends]
        - **Technological**: [Technology impact]
        - **Environmental**: [Environmental considerations]
        - **Legal**: [Legal/regulatory factors]
        ### Strategic Recommendations
        - [Key strategic priorities and actions]
        
        Provide comprehensive analysis with actionable strategic insights.
        """,
    )

    resource_planner = AssistantAgent(
        "ResourcePlanner",
        description="Resource allocation and project planning expert",
        model_client=model_client,
        system_message="""
        You are a Resource Planner focused on practical implementation planning.
        
        Your role is to:
        - Plan resource requirements (human, financial, technical)
        - Create realistic project timelines
        - Estimate costs and budgets
        - Identify critical dependencies
        - Suggest team structure and skills needed
        
        Structure your responses as:
        ## 📋 Resource Planning
        ### Team Requirements
        - [Roles and skills needed]
        ### Timeline & Milestones
        - [Project phases and key milestones]
        ### Budget Estimation
        - [Cost breakdown and financial requirements]
        ### Critical Dependencies
        - [Key dependencies and risk factors]
        ### Implementation Roadmap
        - [Step-by-step execution plan]
        
        Focus on realistic, actionable planning with clear deliverables.
        """,
    )

    success_metrics_agent = AssistantAgent(
        "SuccessMetricsExpert",
        description="KPI definition and success measurement expert",
        model_client=model_client,
        system_message="""
        You are a Success Metrics Expert focused on defining and measuring success.
        
        Your role is to:
        - Define key performance indicators (KPIs)
        - Create measurement frameworks
        - Establish success criteria
        - Design monitoring and evaluation systems
        - Recommend analytics and tracking tools
        
        Structure your responses as:
        ## 🎯 Success Metrics Framework
        ### Key Performance Indicators
        - [Primary KPIs and metrics]
        ### Success Criteria
        - [Clear success definitions]
        ### Measurement Framework
        - [How to track and measure progress]
        ### Monitoring Tools
        - [Recommended analytics and tracking tools]
        ### Review & Optimization
        - [Regular review processes and optimization approaches]
        
        Provide measurable, actionable metrics that drive results.
        """,
    )

    # Create termination conditions
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=40)
    termination = text_mention_termination | max_messages_termination

    # Create brainstorming team
    brainstorm_team = SelectorGroupChat(
        [ideation_agent, inquiry_agent, business_analyst, tech_advisor, strategic_analyst, resource_planner, success_metrics_agent],
        model_client=model_client,
        termination_condition=termination,
        allow_repeated_speaker=True,
        max_turns=20,
    )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent_outputs" not in st.session_state:
        st.session_state.agent_outputs = []
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
        
    if "processed_audio" not in st.session_state:
        st.session_state.processed_audio = set()

    # Main layout
    col_main, col_agents = st.columns([2, 1])
    
    with col_main:
        st.markdown("### 💬 Brainstorming Session")
        st.markdown("*Share your initial idea and let our AI team help you develop it into a comprehensive business opportunity!*")
        
        # Chat history container
        chat_container = st.container(height=450)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    # Play audio if available
                    if message["role"] == "assistant" and "audio_file" in message and message["audio_file"]:
                        st.audio(message["audio_file"], format="audio/mp3")
    
    with col_agents:
        st.markdown("### 🤖 Agent Collaboration")
        st.markdown("*Watch the brainstorming team develop your idea*")
        
        # Agent outputs container
        agent_container = st.container(height=450)
        with agent_container:
            for output in st.session_state.agent_outputs:
                with st.expander(f"🎯 {output['source']}", expanded=False):
                    st.markdown(output['content'])
                    st.caption(f"⏰ {output['timestamp']}")

    # Voice and text input section
    st.markdown("---")
    st.markdown('<div class="voice-controls">', unsafe_allow_html=True)
    
    # Input methods
    col_voice, col_text = st.columns([1, 3])
    
    with col_voice:
        st.markdown("**🎤 Voice Input**")
        audio_data = st.audio_input("Record your idea")
    
    with col_text:
        st.markdown("**⌨️ Text Input**")
        prompt = st.chat_input("Share your business idea or ask for brainstorming help...")

    st.markdown('</div>', unsafe_allow_html=True)
    
    # Example prompts
    st.markdown("### 🚀 Quick Start Ideas")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💡 SaaS Idea"):
            example_prompt = "I want to create a SaaS platform for small businesses to manage their social media presence using AI. Help me brainstorm and develop this idea comprehensively."
            st.session_state.example_prompt = example_prompt
    
    with col2:
        if st.button("🌱 Sustainability Startup"):
            example_prompt = "I'm thinking about a startup that helps consumers reduce their carbon footprint through AI-powered recommendations. Let's brainstorm this concept."
            st.session_state.example_prompt = example_prompt
    
    with col3:
        if st.button("🎓 EdTech Innovation"):
            example_prompt = "I have an idea for personalized learning platform using AI tutors. Help me explore and refine this concept for market readiness."
            st.session_state.example_prompt = example_prompt

    # Handle example prompt
    if hasattr(st.session_state, 'example_prompt') and st.session_state.example_prompt:
        prompt = st.session_state.example_prompt
        delattr(st.session_state, 'example_prompt')

    # Process audio input
    if audio_data and not st.session_state.processing:
        audio_id = str(hash(audio_data.getvalue()))
        
        if audio_id not in st.session_state.processed_audio:
            st.session_state.processing = True
            st.session_state.processed_audio.add(audio_id)
            
            with st.spinner("🎤 Transcribing your idea..."):
                transcription = transcribe_audio(audio_data)
            
            if transcription:
                st.session_state.messages.append({"role": "user", "content": f"🎤 {transcription}"})
                
                with st.spinner("🧠 Brainstorming team is analyzing your idea..."):
                    await process_brainstorm_session(transcription, brainstorm_team)
            
            st.session_state.processing = False
            st.rerun()

    # Process text input
    if prompt and not st.session_state.processing:
        st.session_state.processing = True
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("🧠 Brainstorming team is working on your idea..."):
            await process_brainstorm_session(prompt, brainstorm_team)
        
        st.session_state.processing = False
        st.rerun()

    # Control buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        if st.button("🗑️ Clear Session"):
            st.session_state.messages = []
            st.session_state.agent_outputs = []
            st.session_state.processed_audio = set()
            st.session_state.processing = False
            st.rerun()
    
    with col2:
        if st.button("📊 Export Session"):
            session_export = {
                "timestamp": datetime.datetime.now().isoformat(),
                "session_type": "brainstorming",
                "messages": st.session_state.messages,
                "agent_outputs": st.session_state.agent_outputs
            }
            st.download_button(
                "💾 Download JSON",
                data=str(session_export),
                file_name=f"brainstorm_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("🎯 Generate Summary"):
            if st.session_state.messages:
                summary = generate_session_summary()
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": summary
                })
                st.rerun()
    
    with col4:
        if st.button("ℹ️ Help"):
            st.info("""
            **🧠 AI Brainstorming Studio Guide:**
            
            **How to Use:**
            1. **Share your idea** via voice or text
            2. **Watch the team collaborate** in real-time
            3. **Engage with follow-up questions** to refine ideas
            4. **Get comprehensive analysis** across all business dimensions
            
            **Brainstorming Team:**
            - 🤔 **Ideation Catalyst**: Creative idea generation
            - ❓ **Inquiry Specialist**: Strategic questioning
            - 💼 **Business Analyst**: Market & financial analysis
            - 🚀 **Technology Advisor**: Tech trends & implementation
            - 📊 **Strategic Analyst**: SWOT & PESTEL analysis
            - 📋 **Resource Planner**: Implementation planning
            - 🎯 **Success Metrics Expert**: KPI definition & measurement
            
            **Features:**
            - 🎤 Voice input with transcription
            - 🔊 Audio playback of responses
            - 📊 Real-time agent collaboration
            - 📁 Session export capabilities
            - 🎯 Comprehensive idea development
            """)

async def process_brainstorm_session(user_input, team):
    """Process a brainstorming session with the agent team"""
    try:
        all_agent_responses = []
        
        async for message in team.run_stream(task=user_input):
            # Extract source and content safely
            if hasattr(message, 'source'):
                source = message.source
            elif hasattr(message, 'agent_name'):
                source = message.agent_name
            else:
                source = "System"
                
            if hasattr(message, 'content'):
                content = message.content
                print(f"[{source}]: {content}")
                
                # Store agent output
                st.session_state.agent_outputs.append({
                    'source': source,
                    'content': content,
                    'timestamp': datetime.datetime.now().strftime("%H:%M:%S")
                })
                
                all_agent_responses.append(f"**{source}:**\n{content}")
            else:
                content = str(message)
                print(f"[{source}]: {content}")
                
                st.session_state.agent_outputs.append({
                    'source': source,
                    'content': content,
                    'timestamp': datetime.datetime.now().strftime("%H:%M:%S")
                })
        
        # Create comprehensive summary
        if all_agent_responses:
            summary = create_brainstorm_summary(all_agent_responses)
            
            # Generate audio for summary
            with st.spinner("🔊 Generating audio summary..."):
                audio_file = generate_audio_response(summary[:1000])  # Limit for audio
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": summary,
                "audio_file": audio_file
            })
        
    except Exception as e:
        st.error(f"❌ An error occurred during brainstorming: {str(e)}")
        print(f"Error: {e}")

def create_brainstorm_summary(agent_responses):
    """Create a comprehensive summary from all agent responses"""
    summary = f"""
# 🧠 Comprehensive Brainstorming Analysis

## 📋 Executive Summary
Our AI brainstorming team has analyzed your idea from multiple perspectives to provide a comprehensive development framework.

---

## 🔍 Multi-Perspective Analysis

{chr(10).join(agent_responses)}

---

## 🎯 Key Takeaways & Next Steps

### Immediate Actions
1. **Validate Core Assumptions**: Test key hypotheses with target customers
2. **Competitive Research**: Deep dive into competitive landscape
3. **Technical Feasibility**: Conduct proof-of-concept development
4. **Market Validation**: Interview potential customers and stakeholders

### Medium-term Goals
1. **MVP Development**: Build minimum viable product
2. **Team Building**: Recruit key personnel based on resource plan
3. **Funding Strategy**: Prepare for investment rounds if needed
4. **Partnership Development**: Establish strategic partnerships

### Long-term Vision
1. **Scale Operations**: Expand based on success metrics
2. **Market Expansion**: Enter adjacent markets or geographies
3. **Innovation Pipeline**: Develop next-generation features
4. **Exit Strategy**: Plan for acquisition or IPO opportunities

---

## 💡 Innovation Opportunities
- Leverage emerging technologies for competitive advantage
- Explore partnership and collaboration opportunities
- Consider platform or ecosystem approaches
- Investigate sustainability and social impact angles

## 📊 Success Indicators
- Clear KPIs and measurement frameworks established
- Regular review and optimization processes in place
- Strong market feedback and customer validation
- Sustainable business model with clear value proposition

---

*This comprehensive analysis provides a 360-degree view of your idea's potential. Use these insights to refine your concept and create a compelling business opportunity.*
"""
    return summary

def generate_session_summary():
    """Generate a summary of the current brainstorming session"""
    if not st.session_state.messages:
        return "No session content to summarize."
    
    return f"""
# 📊 Brainstorming Session Summary

**Session Date**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 🎯 Session Overview
- **Total Messages**: {len(st.session_state.messages)}
- **Agent Interactions**: {len(st.session_state.agent_outputs)}
- **Active Agents**: {len(set(output['source'] for output in st.session_state.agent_outputs))}

## 💡 Key Ideas Explored
{chr(10).join([f"- {msg['content'][:100]}..." for msg in st.session_state.messages if msg['role'] == 'user'])}

## 🤖 Agent Contributions
{chr(10).join([f"- **{output['source']}**: {output['content'][:150]}..." for output in st.session_state.agent_outputs[-5:]])}

## 🚀 Recommended Next Steps
1. Review all agent recommendations
2. Prioritize actionable insights
3. Develop implementation timeline
4. Schedule follow-up brainstorming sessions

*Continue the conversation to further refine and develop your ideas!*
"""

if __name__ == "__main__":
    import asyncio
    asyncio.run(brainstorm_assistant())
