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
# client = AzureOpenAI(
#     azure_endpoint=endpoint,
#     api_key=api_key,
#     api_version="2024-10-21",  # Adjust API version as needed
# )

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"   # audience / scope for Azure OpenAI
)

client = AzureOpenAI(
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
#api_key=os.getenv("AZURE_OPENAI_KEY"),  
azure_ad_token_provider=token_provider,  # Use Azure AD authentication
api_version="2024-10-21",
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
    temperature=0.1,
)

async def fine_tuning_pipeline_assistant():
    
    st.set_page_config(
        page_title="🎯 Fine-tuning Pipeline Agent Chat",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .agent-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .task-badge {
        display: inline-block;
        background: #e1f5fe;
        color: #01579b;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        margin: 0.1rem;
    }
    .stExpander > div > div > div > div {
        background-color: #f8f9fa;
    }
    .section-header {
        background: linear-gradient(90deg, #e3f2fd, #f3e5f5);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .section-content {
        background: #fafafa;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🎯 Fine-tuning Pipeline Agent Chat System</h1>', unsafe_allow_html=True)
    
    # Sidebar with agent information
    with st.sidebar:
        st.markdown("## 🎯 Available Agents")
        
        agent_info = {
            "🎯 Use Case Agent": {
                "tasks": ["T1.2 (onboard use case)", "T1.3 (validate feasibility)"],
                "function": "Extracts use case details, logs them in project management tools, and performs technical feasibility checks",
                "tools": "Jira/Airtable, data catalogs, compliance databases"
            },
            "📊 Data Agent": {
                "tasks": ["T1.4 (identify data sources)", "T1.5 (collect/generate data)", "T2.1 (preprocess data)", "T4.1 (expand dataset)", "T4.2 (re-preprocess)"],
                "function": "Scans data sources, collects/generates data, and automates preprocessing",
                "tools": "Internal/external data sources, pandas/NLTK/OpenCV"
            },
            "🧠 Model Agent": {
                "tasks": ["T2.2 (select model)", "T2.3 (fine-tune model)", "T3.1 (evaluate model)", "T4.3 (optimize/retrain)"],
                "function": "Evaluates model architectures, manages fine-tuning, optimizes hyperparameters",
                "tools": "Hugging Face, PyTorch/TensorFlow, scikit-learn"
            },
            "⚙️ Pipeline Agent": {
                "tasks": ["T5.1 (design API)", "T5.2 (optimize model)", "T5.3 (integrate pipeline)", "T6.1 (deploy to staging)", "T6.2 (end-to-end testing)", "T8.1 (deploy to production)"],
                "function": "Generates API templates, optimizes models for deployment, automates integration testing",
                "tools": "FastAPI, ONNX Runtime, Kubernetes, pytest, Locust"
            },
            "📈 Monitoring Agent": {
                "tasks": ["T8.2 (initial monitoring)", "T9.1 (set up dashboards/alerts)", "T9.2 (retraining triggers)"],
                "function": "Monitors model performance, sets up dashboards/alerts, triggers retraining",
                "tools": "Prometheus, Grafana, PagerDuty, MLflow"
            },
            "👨‍💼 Human Oversight": {
                "tasks": ["T1.1, T3.2, T6.3, T7.1 (human validation tasks)"],
                "function": "Provides dashboards for human review, approves agent outputs, handles escalations",
                "tools": "Slack, Confluence, Matplotlib, Tableau"
            }
        }
        
        for agent_name, info in agent_info.items():
            with st.expander(agent_name):
                st.markdown(f"**Function:** {info['function']}")
                st.markdown("**Tasks:**")
                for task in info['tasks']:
                    st.markdown(f"- {task}")
                st.markdown(f"**Tools:** {info['tools']}")

    # Create agents with enhanced system messages for fine-tuning focus
    use_case_agent = AssistantAgent(
        "UseCaseAgent",
        description="An agent for use case onboarding and feasibility validation for fine-tuning projects",
        model_client=model_client,
        system_message="""
        You are a Use Case Agent responsible for fine-tuning pipeline projects:
        - T1.2: Onboarding use cases by extracting and documenting fine-tuning requirements
        - T1.3: Validating technical feasibility for model fine-tuning compatibility
        
        Your capabilities include:
        - Interfacing with project management tools (Jira/Airtable)
        - Accessing data catalogs and compliance databases
        - Performing technical feasibility assessments for fine-tuning projects
        - Documenting use case requirements and constraints for model customization
        
        When providing responses, structure them as:
        ## 📋 Use Case Analysis
        ### Requirements Gathering
        - [Detail specific requirements]
        ### Feasibility Assessment  
        - [Provide technical feasibility analysis]
        ### Recommendations
        - [Clear actionable recommendations]
        
        Always provide detailed analysis for fine-tuning viability and customization needs.
        """,
    )

    data_agent = AssistantAgent(
        "DataAgent",
        description="An agent for data identification, collection, and preprocessing for fine-tuning",
        model_client=model_client,
        system_message="""
        You are a Data Agent responsible for fine-tuning data preparation:
        - T1.4: Identifying relevant data sources for fine-tuning datasets
        - T1.5: Collecting and generating fine-tuning data
        - T2.1: Preprocessing data (cleaning, formatting, augmentation for fine-tuning)
        - T4.1: Expanding datasets when needed for better fine-tuning results
        - T4.2: Re-preprocessing data based on fine-tuning requirements
        
        Your capabilities include:
        - Scanning internal and external data sources
        - Using pandas/NLTK/OpenCV for fine-tuning data processing
        - Implementing data quality checks and validation
        - Managing fine-tuning data versioning and lineage
        
        When providing responses, structure them as:
        ## 📊 Data Strategy
        ### Data Source Identification
        - [List potential data sources]
        ### Data Collection Plan
        - [Detail collection methodology]
        ### Preprocessing Pipeline
        - [Outline preprocessing steps]
        ### Quality Assurance
        - [Data validation and quality measures]
        
        Focus on data quality, compliance, and optimal preprocessing for fine-tuning success.
        """,
    )

    model_agent = AssistantAgent(
        "ModelAgent",
        description="An agent for model selection, fine-tuning, and optimization",
        model_client=model_client,
        system_message="""
        You are a Model Agent responsible for fine-tuning operations:
        - T2.2: Selecting appropriate base models for fine-tuning
        - T2.3: Fine-tuning models for specific use cases and domains
        - T3.1: Evaluating fine-tuned model performance and metrics
        - T4.3: Optimizing and re-fine-tuning models
        
        Your capabilities include:
        - Interfacing with model zoos (Hugging Face) for base model selection
        - Managing fine-tuning pipelines (PyTorch/TensorFlow)
        - Using evaluation libraries (scikit-learn) for performance assessment
        - Hyperparameter optimization for fine-tuning processes
        
        When providing responses, structure them as:
        ## 🧠 Model Fine-tuning Strategy
        ### Base Model Selection
        - [Recommend suitable base models]
        ### Fine-tuning Approach
        - [Detail fine-tuning methodology]
        ### Hyperparameter Configuration
        - [Specify optimal hyperparameters]
        ### Evaluation Metrics
        - [Define success metrics and evaluation criteria]
        
        Provide detailed model recommendations with fine-tuning performance justifications.
        """,
    )

    pipeline_agent = AssistantAgent(
        "PipelineAgent",
        description="An agent for fine-tuning pipeline design, optimization, and deployment",
        model_client=model_client,
        system_message="""
        You are a Pipeline Agent responsible for fine-tuning pipeline operations:
        - T5.1: Designing APIs and service interfaces for fine-tuned models
        - T5.2: Optimizing fine-tuned models for deployment
        - T5.3: Integrating fine-tuning pipelines into production systems
        - T6.1: Deploying fine-tuned models to staging environments
        - T6.2: Conducting end-to-end testing of fine-tuned systems
        - T8.1: Deploying fine-tuned models to production
        
        Your capabilities include:
        - Using FastAPI for fine-tuned model API development
        - ONNX Runtime for fine-tuned model optimization
        - Kubernetes for fine-tuning pipeline orchestration
        - Testing frameworks (pytest, Locust) for fine-tuned model validation
        
        When providing responses, structure them as:
        ## ⚙️ Pipeline Architecture
        ### API Design
        - [Detail API structure for fine-tuned models]
        ### Deployment Strategy
        - [Outline deployment approach]
        ### Integration Plan
        - [Integration with existing systems]
        ### Testing Framework
        - [Comprehensive testing strategy]
        
        Focus on scalability, reliability, and deployment best practices for fine-tuned models.
        """,
    )

    monitoring_agent = AssistantAgent(
        "MonitoringAgent",
        description="An agent for monitoring fine-tuned models, alerting, and retraining triggers",
        model_client=model_client,
        system_message="""
        You are a Monitoring Agent responsible for fine-tuned model monitoring:
        - T8.2: Setting up monitoring systems for fine-tuned models
        - T9.1: Creating dashboards and alerts for fine-tuned model performance
        - T9.2: Implementing retraining triggers based on fine-tuned model drift
        
        Your capabilities include:
        - Integrating with Prometheus and Grafana for fine-tuned model monitoring
        - Setting up PagerDuty alerts for fine-tuned model issues
        - Using MLflow for fine-tuning experiment tracking
        - Monitoring fine-tuned model performance and data drift
        
        When providing responses, structure them as:
        ## 📈 Monitoring Strategy
        ### Performance Metrics
        - [Define key performance indicators]
        ### Alert Configuration
        - [Setup alerting thresholds and notifications]
        ### Dashboard Design
        - [Monitoring dashboard specifications]
        ### Retraining Triggers
        - [Criteria for model retraining]
        
        Provide comprehensive monitoring strategies for fine-tuned model lifecycle management.
        """,
    )

    human_oversight_agent = AssistantAgent(
        "HumanOversightAgent",
        description="An agent for human validation and oversight coordination in fine-tuning projects",
        model_client=model_client,
        system_message="""
        You are a Human Oversight Agent responsible for fine-tuning project oversight:
        - T1.1, T3.2, T6.3, T7.1: Coordinating human validation tasks for fine-tuning
        - Providing dashboards for human review and SME validation of fine-tuned models
        - Approving fine-tuning agent outputs and handling escalations
        - Managing medium/low suitability tasks requiring human input in fine-tuning projects
        
        Your capabilities include:
        - Interfacing with collaboration platforms (Slack, Confluence)
        - Using visualization tools (Matplotlib, Tableau) for fine-tuning insights
        - Coordinating human-in-the-loop workflows for fine-tuning validation
        - Managing approval processes and escalations for fine-tuning projects
        
        When providing responses, structure them as:
        ## 👨‍💼 Human Oversight Framework
        ### Validation Requirements
        - [Human validation checkpoints]
        ### Review Process
        - [SME review and approval workflow]
        ### Escalation Procedures
        - [Issue escalation and resolution]
        ### Quality Assurance
        - [Human oversight quality measures]
        
        Focus on human-AI collaboration and ensuring quality oversight in fine-tuning projects.
        """,
    )

    # Create termination conditions
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=30)
    termination = text_mention_termination | max_messages_termination

    # Create team
    team = SelectorGroupChat(
        [use_case_agent, data_agent, model_agent, pipeline_agent, monitoring_agent, human_oversight_agent],
        model_client=model_client,
        termination_condition=termination,
        allow_repeated_speaker=True,
        max_turns=15,
    )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "agent_outputs" not in st.session_state:
        st.session_state.agent_outputs = []
    
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # Create layout with columns
    col_main, col_agents = st.columns([2, 1])
    
    with col_main:
        st.markdown("### 💬 Fine-tuning Pipeline Chat")
        st.markdown("*Describe your fine-tuning task and watch the agents collaborate to create a complete solution!*")
        
        # Chat history container
        chat_container = st.container(height=500)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
    
    with col_agents:
        st.markdown("### 🤖 Agent Outputs")
        st.markdown("*Real-time agent collaboration details*")
        
        # Agent outputs container
        agent_container = st.container(height=500)
        with agent_container:
            for output in st.session_state.agent_outputs:
                with st.expander(f"🎯 {output['source']}", expanded=False):
                    st.markdown(output['content'])
                    st.caption(f"⏰ {output['timestamp']}")

    # Input section
    st.markdown("---")
    
    # Example prompts
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🎯 Fine-tune: Medical Image Analysis"):
            example_prompt = "I need to fine-tune a vision transformer model for medical X-ray analysis to detect pneumonia. Guide me through the complete fine-tuning pipeline from data preparation to production deployment."
            st.session_state.example_prompt = example_prompt
    
    with col2:
        if st.button("📊 Fine-tune: Domain-Specific NLP"):
            example_prompt = "Help me fine-tune a BERT model for legal document classification. I have legal case documents but need guidance on data preprocessing, fine-tuning strategy, and performance monitoring."
            st.session_state.example_prompt = example_prompt
    
    with col3:
        if st.button("🚀 Fine-tune: Custom LLM"):
            example_prompt = "I want to fine-tune a large language model for customer service chatbot. Guide me through model selection, fine-tuning approach, API deployment, and monitoring setup."
            st.session_state.example_prompt = example_prompt

    # Chat input
    prompt = st.chat_input("I am looking to create a HR onboarding system to onboard new employees with companies code of conduit and procedures? (e.g., 'Fine-tune GPT for code generation')...")
    
    # Handle example prompt
    if hasattr(st.session_state, 'example_prompt') and st.session_state.example_prompt:
        prompt = st.session_state.example_prompt
        delattr(st.session_state, 'example_prompt')

    # Process user input
    if prompt and not st.session_state.processing:
        st.session_state.processing = True
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("🎯 Agents are collaborating on your fine-tuning pipeline...", show_time=True):
            try:
                last_message = None
                async for message in team.run_stream(task=prompt):
                    print(f"Message type: {type(message)}")
                    print(f"Message attributes: {dir(message)}")
                    
                    # Try to extract source and content safely
                    if hasattr(message, 'source'):
                        source = message.source
                    elif hasattr(message, 'agent_name'):
                        source = message.agent_name
                    else:
                        source = "System"
                        
                    if hasattr(message, 'content'):
                        content = message.content
                        print(f"[{source}]: {content}")
                        last_message = content
                        
                        # Store agent output
                        st.session_state.agent_outputs.append({
                            'source': source,
                            'content': content,
                            'timestamp': datetime.datetime.now().strftime("%H:%M:%S")
                        })
                    else:
                        content = str(message)
                        print(f"[{source}]: {content}")
                        
                        # Store raw message
                        st.session_state.agent_outputs.append({
                            'source': source,
                            'content': content,
                            'timestamp': datetime.datetime.now().strftime("%H:%M:%S")
                        })
                
                # Add final response with comprehensive summary
                if last_message:
                    # Create a comprehensive summary structure
                    summary_content = f"""
# 🎯 Complete Fine-tuning Pipeline Solution

## 📋 Executive Summary
{last_message}

---

## 🔄 Fine-tuning Pipeline Overview

### Phase 1: Use Case & Requirements Analysis
- **Objective**: Define fine-tuning goals and validate feasibility
- **Key Activities**: Requirements gathering, technical assessment, resource planning
- **Deliverables**: Use case specification, feasibility report, project roadmap

### Phase 2: Data Strategy & Preparation  
- **Objective**: Prepare high-quality fine-tuning datasets
- **Key Activities**: Data collection, cleaning, formatting, augmentation
- **Deliverables**: Training dataset, validation dataset, data quality report

### Phase 3: Model Selection & Fine-tuning
- **Objective**: Select optimal base model and execute fine-tuning
- **Key Activities**: Base model evaluation, hyperparameter tuning, training execution
- **Deliverables**: Fine-tuned model, performance metrics, training logs

### Phase 4: Pipeline Integration & Testing
- **Objective**: Integrate fine-tuned model into production pipeline
- **Key Activities**: API development, system integration, comprehensive testing
- **Deliverables**: Production-ready API, test results, deployment package

### Phase 5: Deployment & Monitoring
- **Objective**: Deploy to production with comprehensive monitoring
- **Key Activities**: Production deployment, monitoring setup, alert configuration
- **Deliverables**: Live system, monitoring dashboards, operational procedures

### Phase 6: Human Oversight & Quality Assurance
- **Objective**: Ensure quality and manage ongoing improvements
- **Key Activities**: Performance review, stakeholder validation, continuous improvement
- **Deliverables**: Quality reports, approval workflows, improvement recommendations

---

## 📊 Success Metrics & KPIs

### Technical Metrics
- Model accuracy/performance improvements
- Inference latency and throughput
- Resource utilization efficiency
- Data quality scores

### Business Metrics  
- User satisfaction scores
- Cost per inference
- Time to deployment
- ROI on fine-tuning investment

### Operational Metrics
- System uptime and reliability
- Alert response times
- Model drift detection accuracy
- Retraining frequency and success rate

---

## 🚀 Next Steps & Recommendations

1. **Immediate Actions**: Begin with use case validation and data assessment
2. **Resource Planning**: Allocate computational resources for fine-tuning
3. **Timeline**: Establish realistic milestones for each phase
4. **Risk Mitigation**: Identify potential challenges and mitigation strategies
5. **Stakeholder Alignment**: Ensure all stakeholders understand the approach

---

*This comprehensive solution has been collaboratively designed by all specialized agents to ensure success in your fine-tuning project.*
"""
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": summary_content
                    })
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                print(f"Error: {e}")
        
        st.session_state.processing = False
        st.rerun()

    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.agent_outputs = []
            st.session_state.processing = False
            st.rerun()
    
    with col2:
        if st.button("📊 Export Chat"):
            chat_export = {
                "timestamp": datetime.datetime.now().isoformat(),
                "messages": st.session_state.messages,
                "agent_outputs": st.session_state.agent_outputs
            }
            st.download_button(
                "💾 Download JSON",
            data=str(chat_export),
            file_name=f"fine_tuning_pipeline_chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col3:
        if st.button("ℹ️ Help"):
            st.info("""
            **How to use this Fine-tuning Pipeline Agent Chat:**
            
            1. **Describe your fine-tuning task** in the chat input
            2. **Watch agents collaborate** in real-time in the right panel
            3. **Review the comprehensive solution** in the main chat area
            4. **Use example buttons** for quick start scenarios
            
            **Available Agents:**
            - 🎯 Use Case Agent: Requirements & feasibility for fine-tuning
            - 📊 Data Agent: Fine-tuning data collection & preprocessing  
            - 🧠 Model Agent: Model selection & fine-tuning execution
            - ⚙️ Pipeline Agent: Deployment & integration of fine-tuned models
            - 📈 Monitoring Agent: Fine-tuned model performance monitoring
            - 👨‍💼 Human Oversight: Quality assurance & validation
            
            **Fine-tuning Pipeline Phases:**
            1. Use Case Analysis & Requirements
            2. Data Strategy & Preparation
            3. Model Selection & Fine-tuning
            4. Pipeline Integration & Testing
            5. Deployment & Monitoring
            6. Human Oversight & Quality Assurance
            """)

if __name__ == "__main__":
    import asyncio
    asyncio.run(fine_tuning_pipeline_assistant())
