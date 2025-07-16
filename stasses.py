import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import tempfile
import uuid
from openai import AzureOpenAI
import streamlit as st
import asyncio
import io
import os
import time
import json
import numpy as np
from datetime import datetime
from typing import Any, Callable, Set, Dict, List, Optional


from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI configuration (replace with your credentials)
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
WHISPER_DEPLOYMENT_NAME = "whisper"
CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version="2024-06-01"  # Adjust API version as needed
)

# --- Load Assessment Configuration from JSON ---
import json
import os
json_path = os.path.join(os.path.dirname(__file__), "aiassessment.json")
with open(json_path, "r", encoding="utf-8") as f:
    DIMENSIONS = json.load(f)

QUADRANTS = [
    ("Strategy & Vision", "Focus on defining/refining AI strategy and governance framework."),
    ("Enablement & Foundation", "Invest in data, tools, and talent development."),
    ("Execution & Scaling", "Scale existing AI initiatives and improve execution."),
    ("Results & Optimization", "Focus on ROI measurement and responsible AI practices."),
]

def aoai_callback(query: str) -> str:
    returntxt = ""
    excelsheetinfo = ""

    client = AzureOpenAI(
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version="2024-12-01-preview",
    )
    system_prompt = (
        """You are a AI Assesment assistant, Process the user query and provide a detailed response.
        Based on the scores between 1 to 5 where 1 is the lowest and 5 is the highest.
        Provide detail recommendation on a strategy to improve the AI maturity of the organization.
        Provide a section on what use cases to start with based on the assesment output provided.
        Can you split the use cases for Generative AI, Vision, Audio and Traditional Deep and Machine Learning, including Time Series Forecasting.
        Provide Step by step on guidance to implement the recommendations.
        """
    )
    txtmessages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
        ]
    model_name = os.getenv("MODEL_DEPLOYMENT_NAME")
    # model_name_reasoning = "o3"

    response = client.chat.completions.create(
        model=model_name,
        #reasoning={"effort": "high"},
        # reasoning_effort="high",
        messages=txtmessages,
        temperature=0.7,
        max_tokens=4000
    )
    # print("Response received from Azure OpenAI:", response)
    # returntxt = response.output_text.strip()
    returntxt = response.choices[0].message.content
    return returntxt

def assesmentmain():
    # --- Streamlit App ---
    # Remove page config when called from main app
    # st.set_page_config(page_title="AI Maturity Assessment", layout="wide")
    st.title("🤖 AI Maturity Assessment Tool")
    st.write("Evaluate your organization's AI maturity across key dimensions and get tailored recommendations.")

    with st.expander("ℹ️ About this Assessment"):
        st.markdown("""
        This tool helps organizations assess their AI maturity across six key dimensions:
        - **Strategy & Governance**
        - **Data & Infrastructure**
        - **Technology & Tools**
        - **Skills & Culture**
        - **Results & Impact**
        - **Responsible AI & Trustworthiness**
        After completing the assessment, you'll see your results plotted on a 4-quadrant model and receive actionable recommendations.
        """)

    # --- Assessment Form ---

    st.header("Step 1: Complete the Assessment")
    # Use a form to collect all answers before processing
    with st.form("ai_assessment_form"):
        scores = {}
        slider_values = {}
        for dim in DIMENSIONS:
            st.subheader(dim["name"])
            dim_scores = []
            for idx, q in enumerate(dim["questions"]):
                col_q, col_s = st.columns([3, 2])
                with col_q:
                    st.markdown(f"**{q['text']}**<br/><span style='font-size:0.9em;color:gray'>{q['desc']}</span>", unsafe_allow_html=True)
                with col_s:
                    slider_key = f"{dim['name']}_{idx}"
                    val = st.slider(" ", 1, 5, 3, key=slider_key)
                    dim_scores.append(val)
                    slider_values[slider_key] = val
            scores[dim["name"]] = np.mean(dim_scores)
        process = st.form_submit_button("Process Assessment")


    with st.spinner("Processing your assessment...", show_time=True):
        
        # Only process and show results if the button is pressed
        if 'process' in locals() and process:
            # --- Weighted Score Calculation ---
            weighted_scores = {dim["name"]: scores[dim["name"]] * dim["weight"] for dim in DIMENSIONS}
            total_score = sum(weighted_scores.values())

            # --- Quadrant Calculation ---
            # X-axis: Results & Impact, Y-axis: Strategy & Governance
            x = scores["Results & Impact"]
            y = scores["Strategy & Governance"]
            bubble_size = int(30 + 20 * np.mean([scores["Data & Infrastructure"], scores["Technology & Tools"], scores["Skills & Culture"]]))
            color_score = scores["Responsible AI & Trustworthiness"]

            # --- Quadrant Assignment ---
            if x < 3 and y >= 3:
                quadrant = 2
            elif x >= 3 and y < 3:
                quadrant = 3
            elif x < 3 and y < 3:
                quadrant = 1
            else:
                quadrant = 4

            # --- Quadrant Chart ---
            st.header("Step 2: Visualize Your AI Maturity")
            fig = px.scatter(
                x=[x], y=[y],
                size=[bubble_size],
                color=[color_score],
                color_continuous_scale=['red', 'yellow', 'green'],
                range_x=[1, 5], range_y=[1, 5],
                labels={'x': 'Results & Impact', 'y': 'Strategy & Governance'},
                title="AI Maturity Quadrant"
            )
            fig.add_shape(type="rect", x0=1, y0=1, x1=3, y1=3, line=dict(color="gray", width=1))
            fig.add_shape(type="rect", x0=3, y0=1, x1=5, y1=3, line=dict(color="gray", width=1))
            fig.add_shape(type="rect", x0=1, y0=3, x1=3, y1=5, line=dict(color="gray", width=1))
            fig.add_shape(type="rect", x0=3, y0=3, x1=5, y1=5, line=dict(color="gray", width=1))
            fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # --- Results Table ---
            st.header("Step 3: Results & Recommendations")
            col1, col2 = st.columns([2, 3])
            with col1:
                st.markdown("### Your Scores")
                df = pd.DataFrame({
                    "Dimension": [dim["name"] for dim in DIMENSIONS],
                    "Score (1-5)": [round(scores[dim["name"]], 2) for dim in DIMENSIONS],
                    "Weighted": [round(weighted_scores[dim["name"]], 2) for dim in DIMENSIONS],
                })
                st.dataframe(df, use_container_width=True)

            with col2:
                st.markdown("### Quadrant & Recommendations")
                q_name, q_reco = QUADRANTS[quadrant-1]
                st.markdown(f"**Quadrant {quadrant}: {q_name}**")
                st.info(q_reco)
                st.markdown("** Recommendation:**")
                # --- Show all questions and selected values at the bottom ---
                qnatext = ""
                if 'process' in locals() and process:
                    st.markdown("---")
                    # st.header("All Questions and Your Selected Values")
                    for dim in DIMENSIONS:
                        # st.subheader(dim["name"])
                        qnatext += f"### {dim['name']}\n"
                        for idx, q in enumerate(dim["questions"]):
                            slider_key = f"{dim['name']}_{idx}"
                            val = slider_values.get(slider_key, None)
                            # st.write(f"**Q:** {q['text']}")
                            # st.write(f"**Selected Value:** {val if val is not None else 'N/A'}")
                            qnatext += f"- {q['text']}: {val if val is not None else 'N/A'}\n"

                if qnatext:
                    resultrs = aoai_callback(qnatext)
                    st.markdown("### AI Assistant Recommendations")
                    st.write(resultrs)

            st.success("Assessment complete! Use this roadmap to guide your AI journey. You can adjust your answers and see how your quadrant and recommendations change.")

if __name__ == "__main__":
    assesmentmain()