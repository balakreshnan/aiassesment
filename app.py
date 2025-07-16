import streamlit as st
import asyncio
import io
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any
from sahome import sa_assist
from stasses import assesmentmain
from stfinetuneasses import finetuneassesment

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Streamlit page with Material Design 3 theme
st.set_page_config(
    page_title="Solution Architect Assist",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
nav_option = st.sidebar.selectbox("Navigation", ["Home", 
                                                "Solution Architect Assist",
                                                "AI Assesment",
                                                "Fine-tune Assesment",
                                                "About"])

# Display the selected page
if nav_option == "Home":
    asyncio.run(sa_assist())
elif nav_option == "Solution Architect Assists":
    asyncio.run(sa_assist())
elif nav_option == "Fine-tune Assesment":
    finetuneassesment()
elif nav_option == "AI Assesment":
    assesmentmain()
else:
    asyncio.run(sa_assist())