import streamlit as st
import json
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI configuration
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version="2024-06-01"
)

# Load the JSON questionnaire
def load_questionnaire():
    try:
        with open("fine_tuning_questionnaire.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Create a default questionnaire if file doesn't exist
        default_questionnaire = {
            "sections": [
                {
                    "title": "Use Case & Requirements",
                    "questions": [
                        {
                            "id": "q1",
                            "text": "What is your primary use case for fine-tuning?",
                            "type": "multiple_choice",
                            "options": [
                                "Text classification",
                                "Text generation",
                                "Question answering",
                                "Summarization",
                                "Code generation",
                                "Domain-specific chat",
                                "Other"
                            ]
                        },
                        {
                            "id": "q2",
                            "text": "Do you have a specific dataset for training?",
                            "type": "boolean",
                            "options": ["Yes", "No"]
                        },
                        {
                            "id": "q3",
                            "text": "What is the size of your training dataset?",
                            "type": "multiple_choice",
                            "options": [
                                "Less than 100 examples",
                                "100-1,000 examples",
                                "1,000-10,000 examples",
                                "More than 10,000 examples"
                            ]
                        },
                        {
                            "id": "q4",
                            "text": "What is your target domain?",
                            "type": "multiple_choice",
                            "options": [
                                "Healthcare",
                                "Finance",
                                "Legal",
                                "Education",
                                "Technology",
                                "Customer Service",
                                "Other"
                            ]
                        },
                        {
                            "id": "q5",
                            "text": "What is your expected performance improvement?",
                            "type": "multiple_choice",
                            "options": [
                                "Better accuracy on domain-specific tasks",
                                "Reduced hallucination",
                                "Faster response times",
                                "Better understanding of context",
                                "Consistent output format"
                            ]
                        }
                    ]
                },
                {
                    "title": "Technical Requirements",
                    "questions": [
                        {
                            "id": "q6",
                            "text": "What is your technical expertise level?",
                            "type": "multiple_choice",
                            "options": [
                                "Beginner",
                                "Intermediate",
                                "Advanced",
                                "Expert"
                            ]
                        },
                        {
                            "id": "q7",
                            "text": "Do you have GPU resources available?",
                            "type": "boolean",
                            "options": ["Yes", "No"]
                        },
                        {
                            "id": "q8",
                            "text": "What is your budget for fine-tuning?",
                            "type": "multiple_choice",
                            "options": [
                                "Under $1,000",
                                "$1,000-$10,000",
                                "$10,000-$100,000",
                                "Over $100,000"
                            ]
                        },
                        {
                            "id": "q9",
                            "text": "What is your timeline for deployment?",
                            "type": "multiple_choice",
                            "options": [
                                "Within 1 week",
                                "1-4 weeks",
                                "1-3 months",
                                "More than 3 months"
                            ]
                        }
                    ]
                }
            ]
        }
        return default_questionnaire

def get_ai_guidance(task_description, responses):
    """Get AI guidance on fine-tuning approach based on responses"""
    
    # Format responses for the AI
    formatted_responses = []
    for section in questionnaire["sections"]:
        section_title = section.get("title", "Questions")
        formatted_responses.append(f"\n**{section_title}:**")
        for question in section["questions"]:
            if question["id"] in responses:
                response = responses[question["id"]]
                if isinstance(response, list):
                    response = ", ".join(response)
                formatted_responses.append(f"- {question['text']}: {response}")
    
    responses_text = "\n".join(formatted_responses)
    retturntxt = ""
    
    prompt = f"""
    Based on the following fine-tuning questionnaire responses and task description, provide detailed guidance on how to proceed with fine-tuning an AI model.

    **Task Description:** {task_description}

    **Questionnaire Responses:**
    {responses_text}

    Please provide guidance covering:
    1. **Feasibility Assessment**: Is fine-tuning suitable for this use case?
    2. **Recommended Approach**: What type of fine-tuning should be used?
    3. **Model Selection**: Which base model would be most appropriate?
    4. **Data Preparation**: How should the training data be prepared?
    5. **Training Strategy**: What training parameters and techniques should be used?
    6. **Evaluation Metrics**: How should the model be evaluated?
    7. **Deployment Considerations**: What should be considered for deployment?
    8. **Cost and Timeline Estimate**: Realistic expectations for budget and timeline
    9. **Alternative Approaches**: If fine-tuning isn't suitable, what alternatives exist?
    10. **Next Steps**: Specific actionable recommendations

    Please be detailed and practical in your recommendations.
    """
    
    try:
        # response = client.chat.completions.create(
        #     model=CHAT_DEPLOYMENT_NAME,
        #     messages=[
        #         {"role": "system", "content": "You are an expert AI engineer specializing in fine-tuning large language models. Provide detailed, practical guidance based on the user's requirements."},
        #         {"role": "user", "content": prompt}
        #     ],
        #     max_tokens=3500,
        #     temperature=0.7,
        #     seed=42,
        #     logprobs=True,

        # )
        prompt = f"""You are an expert AI engineer specializing in fine-tuning large language models. Provide detailed, practical guidance based on the user's requirements.
        Question: {prompt}
        """
        mcpclient = AzureOpenAI(  
        base_url = os.getenv("AZURE_OPENAI_ENDPOINT") + "/openai/v1/",  
        api_key= os.getenv("AZURE_OPENAI_KEY"),
        api_version="preview"
        )
        response = mcpclient.responses.create(
            model=CHAT_DEPLOYMENT_NAME, # replace with your model deployment name 
            # tools=[
            #     {
            #         "type": "mcp",
            #         "server_label": "github",
            #         "server_url": "https://api.githubcopilot.com/mcp/",
            #         "headers": {
            #             "Authorization": f"Bearer {PAT_TOKEN}",
            #         },
            #         "require_approval": "never",
            #     },
            # ],
            input=prompt,
            max_output_tokens=3500,
            #instructions=transcription,  # "Generate a response using the MCP API tool.",
            
        )
        # retturntxt = response.choices[0].message.content
        retturntxt = response.output_text
        return retturntxt
    except Exception as e:
        return f"Error getting AI guidance: {str(e)}"
    
questionnaire = load_questionnaire()

def finetuneassesment():
    # Check Azure OpenAI configuration (simplified for embedded use)
    if not AZURE_ENDPOINT or not AZURE_API_KEY:
        st.warning("⚠️ Azure OpenAI configuration is incomplete. Some features may not work.")
        st.info("Please set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY environment variables.")

    # Streamlit app
    st.title("🤖 AI Fine-Tuning Guidance Tool")
    st.write("Describe your task and answer the questions below to get personalized fine-tuning guidance from Azure OpenAI GPT-4.")

    # Task description input
    st.header("Step 1: Describe Your Task")
    task_description = st.text_area(
        "Please describe the specific task you want to fine-tune a model for:",
        placeholder="e.g., I want to create a chatbot that can answer questions about my company's HR policies and procedures...",
        height=100
    )

    # Initialize session state
    if "responses" not in st.session_state:
        st.session_state.responses = {}
    if "timestamp" not in st.session_state:
        from datetime import datetime
        st.session_state.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Display all questions in a form
    if questionnaire and "sections" in questionnaire:
        st.header("Step 2: Answer All Questions")
        
        with st.form("fine_tuning_questionnaire"):
            responses = {}
            
            for section in questionnaire["sections"]:
                # Handle missing title key
                section_title = section.get("title", "Questions")
                st.subheader(section_title)
                
                # Skip sections without questions
                if "questions" not in section:
                    continue
                
                for question in section["questions"]:
                    # Skip questions without required fields
                    if "id" not in question or "text" not in question or "type" not in question:
                        continue
                        
                    if question["type"] == "boolean":
                        options = question.get("options", ["Yes", "No"])
                        response = st.radio(
                            question["text"],
                            options,
                            key=question["id"]
                        )
                    elif question["type"] == "multiple_choice":
                        options = question.get("options", [])
                        if not options:
                            st.warning(f"No options provided for question: {question['text']}")
                            continue
                            
                        if len(options) <= 4:
                            response = st.radio(
                                question["text"],
                                options,
                                key=question["id"]
                            )
                        else:
                            response = st.multiselect(
                                question["text"],
                                options,
                                key=question["id"]
                            )
                    
                    responses[question["id"]] = response
            
            submitted = st.form_submit_button("Get AI Guidance")
        
        # Handle form submission outside the form
        if submitted:
            if not task_description.strip():
                st.error("Please describe your task before submitting.")
            else:
                st.session_state.responses = responses
                st.session_state.task_description = task_description
                
                # Get AI guidance
                with st.spinner("Getting personalized fine-tuning guidance from Azure OpenAI..."):
                    guidance = get_ai_guidance(task_description, responses)
                
                st.session_state.guidance = guidance
                returntxt = guidance
        
        # Display results outside the form
        if hasattr(st.session_state, 'guidance') and st.session_state.guidance:
            st.header("Step 3: Personalized Fine-Tuning Guidance")
            st.markdown(st.session_state.guidance)
            
            # Save responses and guidance
            col1, col2 = st.columns(2)
            with col1:
                responses_data = {
                    "task_description": st.session_state.task_description,
                    "responses": st.session_state.responses,
                    "timestamp": st.session_state.get("timestamp", "")
                }
                responses_json = json.dumps(responses_data, indent=2)
                st.download_button(
                    label="Download Responses",
                    data=responses_json,
                    file_name="fine_tuning_responses.json",
                    mime="application/json"
                )
            
            with col2:
                guidance_data = {
                    "task_description": st.session_state.task_description,
                    "guidance": st.session_state.guidance,
                    "timestamp": st.session_state.get("timestamp", "")
                }
                guidance_json = json.dumps(guidance_data, indent=2)
                st.download_button(
                    label="Download Guidance",
                    data=guidance_json,
                    file_name="fine_tuning_guidance.json",
                    mime="application/json"
                )
            
            # Reset option
            if st.button("Start Over"):
                st.session_state.responses = {}
                st.session_state.task_description = ""
                if hasattr(st.session_state, 'guidance'):
                    del st.session_state.guidance
                st.rerun()

    else:
        st.error("Could not load questionnaire. Please check the configuration.")

if __name__ == "__main__":
    finetuneassesment()