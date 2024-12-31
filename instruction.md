## Instructions

### Goal
1. **Build a LLM-powered Streamlit application** that acts as an “AI Data Analyst Agent,” enabling:
   - **Data file upload** and **basic structure/meaning** interpretation.
   - **Data cleaning and manipulation**.
   - **Basic exploratory data analysis** (descriptive statistics & basic visualizations).

2. **LangChain Agent**: Orchestrates user requests and calls specialized “tools” to handle tasks such as data ingestion, column explanation (OpenAI), data cleaning, and basic EDA steps.

3. **LLM Integration**: Use OpenAI’s API for text-based interpretations and guidance, integrated via LangChain.

---

### Overall Project Structure
*(This is a suggested structure; you can adjust it as you see fit. Please keep it simple and iterate quickly.)*

1. **`app.py`** – Main Streamlit application file.  
2. **`requirements.txt`** – Python dependencies (Streamlit, OpenAI, pandas, numpy, matplotlib, seaborn, LangChain, etc.).  
3. **`agents/` folder** – Contains agent logic files (e.g., `ai_data_analyst_agent.py`) that set up the LangChain agent and define tools.  
4. **`tools/` folder** – Each file corresponds to a “tool” (or set of tools) for data ingestion, cleaning, EDA, etc.  
5. **`utils/` folder** – Helper modules (`analysis_utils.py`, `eda_utils.py`) for function definitions.

---

### Step-by-Step Development Instructions

1. **Set up the project and folder structure first.** Keep it simple; we want to iterate quickly.  
2. **Install dependencies**, set up the project environment, and **integrate your OpenAI key** for LLM capability.  
3. **Identify the tools** you will need to implement the project to achieve the goal:
   - Data file upload and basic structure/meaning interpretation.  
   - Data cleaning and manipulation.  
   - Basic exploratory data analysis (descriptive statistics & basic visualizations).  
4. **Build an agent using LangChain** with the tools you identified in the `tools/` folder.  
5. **Build the Streamlit app UI** for user interaction:
   - Let users upload files.  
   - Let them click buttons or enter text instructions that the agent interprets.  
   - Display results in Streamlit (DataFrame, explanations, cleaning feedback, basic stats).

---

### Additional Requirements

1. **Confirm with the user** if you are unsure about their intention.  
2. **Refer to the context** of the conversation to determine the user’s intention.