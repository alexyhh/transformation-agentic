import streamlit as st
from openai import OpenAI
from datetime import datetime
import json
import hashlib
import time

# Page configuration
st.set_page_config(
    page_title="Transformation Assistant",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Authentication credentials
VALID_CREDENTIALS = {
    "admin": hashlib.sha256("transform2024".encode()).hexdigest(),
    "manager": hashlib.sha256("change2024".encode()).hexdigest()
}

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'openai_api_key' not in st.session_state:
    try:
        st.session_state.openai_api_key = st.secrets["OPENAI_API_KEY"]
    except:
        st.session_state.openai_api_key = None

# Agent state tracking
if 'agent_state' not in st.session_state:
    st.session_state.agent_state = {}

def authenticate(username, password):
    """Authenticate user credentials"""
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    return VALID_CREDENTIALS.get(username) == hashed_password

def login_page():
    """Display login page"""
    st.title("🔄 Transformation Management Assistant")
    st.markdown("### Secure Access Portal")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("#### Login to Continue")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Login", use_container_width=True):
                if authenticate(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
        
        with col_b:
            if st.button("Clear", use_container_width=True):
                st.rerun()
        
        st.markdown("---")
        st.info("**Demo Credentials:**\n\nUsername: `admin` | Password: `transform2024`\n\nUsername: `manager` | Password: `change2024`")

def get_openai_client():
    """Get OpenAI client instance"""
    if st.session_state.openai_api_key:
        try:
            return OpenAI(api_key=st.session_state.openai_api_key)
        except Exception as e:
            st.error(f"Error initializing OpenAI client: {str(e)}")
            return None
    return None

class AgentOrchestrator:
    """Manages the agentic workflow"""
    
    def __init__(self, client):
        self.client = client
        self.agents = [
            {"name": "Orchestrator", "icon": "🎯", "status": "pending"},
            {"name": "Data Extraction", "icon": "📊", "status": "pending"},
            {"name": "Risk Analysis", "icon": "⚠️", "status": "pending"},
            {"name": "Stakeholder Analysis", "icon": "👥", "status": "pending"},
            {"name": "Framework Advisor", "icon": "📚", "status": "pending"},
            {"name": "Action Planning", "icon": "📋", "status": "pending"},
            {"name": "Quality Assurance", "icon": "✓", "status": "pending"},
            {"name": "Synthesis", "icon": "🎨", "status": "pending"}
        ]
        self.results = {}
        self.total_start_time = None
        self.agent_times = {}
    
    def update_agent_status(self, agent_name, status, time_taken=None):
        """Update agent status in session state"""
        for agent in self.agents:
            if agent["name"] == agent_name:
                agent["status"] = status
                if time_taken:
                    self.agent_times[agent_name] = time_taken
                break
        st.session_state.agent_state = {
            "agents": self.agents,
            "results": self.results,
            "agent_times": self.agent_times
        }
    
    def call_agent(self, agent_name, system_prompt, user_input, context=""):
        """Call a single agent and track its execution"""
        start_time = time.time()
        self.update_agent_status(agent_name, "running")
        
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            if context:
                messages.append({"role": "user", "content": f"Context from previous agents:\n{context}"})
            messages.append({"role": "user", "content": user_input})
            
            # Stream response for live thoughts
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                stream=True
            )
            
            # Collect streamed response
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    # Update results in real-time for live thoughts
                    self.results[agent_name] = full_response
                    st.session_state.agent_state["results"] = self.results
            
            time_taken = time.time() - start_time
            self.update_agent_status(agent_name, "complete", time_taken)
            self.results[agent_name] = full_response
            
            return full_response
            
        except Exception as e:
            self.update_agent_status(agent_name, "error")
            error_msg = f"Error in {agent_name}: {str(e)}"
            self.results[agent_name] = error_msg
            return error_msg
    
    def run_sequential_analysis(self, user_input, progress_placeholder, thoughts_placeholder):
        """Run the sequential deep analysis pattern"""
        self.total_start_time = time.time()
        
        # Agent 1: Orchestrator
        orchestrator_prompt = """You are an Orchestrator Agent for transformation management analysis.
        Analyze the user's input and create a brief execution plan (2-3 sentences) explaining what aspects 
        you'll focus on and why. Be concise and strategic."""
        
        plan = self.call_agent(
            "Orchestrator",
            orchestrator_prompt,
            user_input
        )
        self.display_progress(progress_placeholder, thoughts_placeholder)
        
        # Agent 2: Data Extraction
        extraction_prompt = """You are a Data Extraction Agent. Parse the input and extract:
        1. Key entities (people, teams, projects, dates)
        2. Sentiment indicators (positive/negative signals)
        3. Timeline information
        4. Main concerns or achievements
        Format as structured bullet points."""
        
        extracted_data = self.call_agent(
            "Data Extraction",
            extraction_prompt,
            user_input,
            f"Execution Plan: {plan}"
        )
        self.display_progress(progress_placeholder, thoughts_placeholder)
        
        # Agent 3: Risk Analysis
        risk_prompt = """You are a Risk Analysis Agent specializing in transformation risks.
        Analyze the structured data for:
        1. Early warning signs (2-3 week prediction window)
        2. Risk severity scoring (1-10)
        3. Dependencies and bottlenecks
        4. Pattern matching against known risk indicators
        Be specific and quantify risks where possible."""
        
        risks = self.call_agent(
            "Risk Analysis",
            risk_prompt,
            user_input,
            f"Extracted Data:\n{extracted_data}"
        )
        self.display_progress(progress_placeholder, thoughts_placeholder)
        
        # Agent 4: Stakeholder Analysis
        stakeholder_prompt = """You are a Stakeholder Analysis Agent focusing on human dynamics.
        Analyze:
        1. Key stakeholders and their positions
        2. Resistance patterns
        3. Communication gaps
        4. Change readiness levels
        5. Power dynamics and influence
        Provide actionable insights about people and relationships."""
        
        stakeholders = self.call_agent(
            "Stakeholder Analysis",
            stakeholder_prompt,
            user_input,
            f"Risk Analysis:\n{risks}\n\nExtracted Data:\n{extracted_data}"
        )
        self.display_progress(progress_placeholder, thoughts_placeholder)
        
        # Agent 5: Framework Advisor
        framework_prompt = """You are a Framework Advisor Agent with expertise in change management frameworks.
        Based on the situation:
        1. Select the most appropriate framework(s): ADKAR, Kotter's 8-Step, Prosci, McKinsey 7-S
        2. Identify which stage/phase the transformation is in
        3. Recommend specific practices from the framework
        4. Explain why this framework fits the situation
        Be practical and framework-specific."""
        
        framework = self.call_agent(
            "Framework Advisor",
            framework_prompt,
            user_input,
            f"Stakeholder Analysis:\n{stakeholders}\n\nRisk Analysis:\n{risks}"
        )
        self.display_progress(progress_placeholder, thoughts_placeholder)
        
        # Agent 6: Action Planning
        action_prompt = """You are an Action Planning Agent creating concrete recommendations.
        Generate:
        1. Top 3 priority actions with urgency levels
        2. Timeline for each action (immediate/short-term/long-term)
        3. Quick wins vs. strategic initiatives
        4. Resource requirements
        5. Success metrics
        Make recommendations specific, measurable, and actionable."""
        
        actions = self.call_agent(
            "Action Planning",
            action_prompt,
            user_input,
            f"Framework Guidance:\n{framework}\n\nStakeholder Analysis:\n{stakeholders}\n\nRisk Analysis:\n{risks}"
        )
        self.display_progress(progress_placeholder, thoughts_placeholder)
        
        # Agent 7: Quality Assurance
        qa_prompt = """You are a Quality Assurance Agent reviewing all findings.
        Check for:
        1. Contradictions between different analyses
        2. Gaps in the analysis
        3. Unaddressed risks
        4. Feasibility of recommendations
        Provide a brief QA summary (3-4 sentences) noting any concerns or confirming quality."""
        
        all_context = f"""
        Risk Analysis: {risks}
        Stakeholder Analysis: {stakeholders}
        Framework Guidance: {framework}
        Action Plan: {actions}
        """
        
        qa_review = self.call_agent(
            "Quality Assurance",
            qa_prompt,
            user_input,
            all_context
        )
        self.display_progress(progress_placeholder, thoughts_placeholder)
        
        # Agent 8: Synthesis
        synthesis_prompt = """You are a Synthesis Agent creating the final comprehensive report.
        Integrate all findings into a coherent, well-structured report with:
        1. Executive Summary (2-3 paragraphs)
        2. Key Findings organized by category
        3. Critical Risks (prioritized)
        4. Recommended Actions (prioritized)
        5. Next Steps
        
        Make it executive-ready, clear, and actionable. Use markdown formatting with headers."""
        
        final_report = self.call_agent(
            "Synthesis",
            synthesis_prompt,
            user_input,
            all_context + f"\n\nQA Review: {qa_review}"
        )
        self.display_progress(progress_placeholder, thoughts_placeholder)
        
        total_time = time.time() - self.total_start_time
        
        return {
            "plan": plan,
            "extracted_data": extracted_data,
            "risks": risks,
            "stakeholders": stakeholders,
            "framework": framework,
            "actions": actions,
            "qa_review": qa_review,
            "final_report": final_report,
            "total_time": total_time
        }
    
    def display_progress(self, progress_placeholder, thoughts_placeholder):
        """Display progress with observability features"""
        with progress_placeholder.container():
            # Observability 1: Visual Agent Pipeline
            st.markdown("### 🔄 Analysis Pipeline")
            
            completed = sum(1 for agent in self.agents if agent["status"] == "complete")
            total = len(self.agents)
            progress_pct = completed / total
            
            # Progress bar (Observability 3)
            st.progress(progress_pct, text=f"Progress: {completed}/{total} agents complete")
            
            # Time metrics (Observability 3)
            if self.total_start_time:
                elapsed = time.time() - self.total_start_time
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Time Elapsed", f"{elapsed:.1f}s")
                with col2:
                    st.metric("Agents Complete", f"{completed}/{total}")
                with col3:
                    if completed > 0 and completed < total:
                        est_remaining = (elapsed / completed) * (total - completed)
                        st.metric("Est. Remaining", f"{est_remaining:.1f}s")
                    else:
                        st.metric("Est. Remaining", "—")
            
            st.markdown("---")
            
            # Agent status cards
            for agent in self.agents:
                status_icon = {
                    "pending": "⏳",
                    "running": "🔄",
                    "complete": "✅",
                    "error": "❌"
                }
                
                icon = status_icon.get(agent["status"], "⏳")
                time_str = ""
                if agent["name"] in self.agent_times:
                    time_str = f" ({self.agent_times[agent['name']]:.1f}s)"
                
                status_text = agent["status"].title()
                if agent["status"] == "running":
                    status_text = "Processing..."
                
                st.markdown(f"{icon} **{agent['icon']} {agent['name']}**: {status_text}{time_str}")
        
        # Observability 4: Live Agent Thoughts
        with thoughts_placeholder.container():
            st.markdown("### 💭 Live Agent Thoughts")
            
            # Show the currently running or most recent agent's output
            for agent in reversed(self.agents):
                if agent["status"] in ["running", "complete"] and agent["name"] in self.results:
                    result = self.results[agent["name"]]
                    if result and not result.startswith("Error"):
                        with st.expander(f"{agent['icon']} {agent['name']} Output", expanded=(agent["status"] == "running")):
                            # Show preview for complete agents, full output for running
                            if agent["status"] == "complete":
                                preview = result[:300] + "..." if len(result) > 300 else result
                                st.markdown(preview)
                            else:
                                st.markdown(result)
                        break

def main_app():
    """Main application interface"""
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # API Key Status
        st.markdown("### 🔑 API Status")
        if st.session_state.openai_api_key:
            st.success("✅ OpenAI Connected")
        else:
            st.error("❌ API Key Missing")
        
        st.markdown("---")
        
        # User info
        st.markdown(f"**Logged in as:** {st.session_state.username}")
        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.chat_history = []
            st.session_state.agent_state = {}
            st.rerun()
        
        st.markdown("---")
        
        # Info
        st.markdown("### ℹ️ About Agentic Analysis")
        st.info("""
        This system uses **8 specialized AI agents** working sequentially:
        
        1. 🎯 Orchestrator
        2. 📊 Data Extraction
        3. ⚠️ Risk Analysis
        4. 👥 Stakeholder Analysis
        5. 📚 Framework Advisor
        6. 📋 Action Planning
        7. ✓ Quality Assurance
        8. 🎨 Synthesis
        
        Watch them work in real-time!
        """)
    
    # Main content
    st.title("🔄 Agentic Transformation Assistant")
    st.markdown("**Multi-agent deep analysis with full observability**")
    st.markdown("---")
    
    # Single input interface
    st.markdown("### 📝 Describe Your Transformation Situation")
    user_input = st.text_area(
        "Provide details about your transformation project, challenges, team dynamics, or concerns:",
        height=200,
        placeholder="Example: Our Q3 digital transformation is 3 months in. Sprint retrospectives show declining velocity from 45 to 32 story points. Technical debt is increasing. The product owner and dev team are clashing over priorities. Middle management seems disengaged...",
        key="agentic_input"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        analyze_button = st.button("🚀 Start Deep Analysis", type="primary", use_container_width=True)
    with col2:
        if st.button("Clear", use_container_width=True):
            st.rerun()
    
    if analyze_button and user_input and st.session_state.openai_api_key:
        client = get_openai_client()
        if client:
            st.markdown("---")
            
            # Create placeholders for real-time updates
            col1, col2 = st.columns([1, 1])
            
            with col1:
                progress_placeholder = st.empty()
            
            with col2:
                thoughts_placeholder = st.empty()
            
            # Run the agentic workflow
            orchestrator = AgentOrchestrator(client)
            results = orchestrator.run_sequential_analysis(
                user_input,
                progress_placeholder,
                thoughts_placeholder
            )
            
            # Display final results
            st.markdown("---")
            st.markdown("## 📊 Final Analysis Report")
            
            # Executive summary at top
            st.markdown(results["final_report"])
            
            # Detailed results in expanders
            st.markdown("---")
            st.markdown("### 🔍 Detailed Agent Outputs")
            
            with st.expander("🎯 Orchestrator - Execution Plan"):
                st.markdown(results["plan"])
            
            with st.expander("📊 Data Extraction - Structured Data"):
                st.markdown(results["extracted_data"])
            
            with st.expander("⚠️ Risk Analysis - Identified Risks"):
                st.markdown(results["risks"])
            
            with st.expander("👥 Stakeholder Analysis - People & Dynamics"):
                st.markdown(results["stakeholders"])
            
            with st.expander("📚 Framework Advisor - Change Management Guidance"):
                st.markdown(results["framework"])
            
            with st.expander("📋 Action Planning - Recommendations"):
                st.markdown(results["actions"])
            
            with st.expander("✓ Quality Assurance - Review Summary"):
                st.markdown(results["qa_review"])
            
            # Summary metrics
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Analysis Time", f"{results['total_time']:.1f}s")
            with col2:
                st.metric("Agents Executed", "8")
            with col3:
                avg_time = results['total_time'] / 8
                st.metric("Avg Time per Agent", f"{avg_time:.1f}s")
            
            # Save to history
            st.session_state.chat_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "type": "Agentic Deep Analysis",
                "input": user_input,
                "output": results["final_report"],
                "total_time": results['total_time']
            })
            
    elif analyze_button and not st.session_state.openai_api_key:
        st.error("⚠️ Please configure your OpenAI API key in Streamlit secrets.")
    elif analyze_button and not user_input:
        st.warning("⚠️ Please provide input for analysis.")
    
    # Analysis History
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("## 📜 Analysis History")
        
        for i, entry in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(f"{entry['type']} - {entry['timestamp']} ({entry.get('total_time', 'N/A')}s)"):
                st.markdown("**Input:**")
                st.text(entry['input'][:200] + "..." if len(entry['input']) > 200 else entry['input'])
                st.markdown("**Analysis:**")
                st.markdown(entry['output'])

def about_page():
    """About Us page content"""
    st.title("ℹ️ About the Transformation Agentic Assistant")
    st.markdown("---")

    st.markdown("""
    ### Project Overview

    The **Transformation Agentic Assistant** is a prototype web application that helps transformation
    managers and leaders make sense of complex situations by turning free-text descriptions into a
    structured, executive-ready analysis.

    Instead of a simple one-shot chatbot, this app demonstrates a **multi-agent workflow**:
    eight specialised AI "agents" work **sequentially** to break down the situation, analyse risks
    and stakeholders, recommend frameworks, and generate an integrated report.

    The app is built as a **Streamlit** web application with:
    - ✅ **Password-protected access** (demo accounts: `admin` / `manager`)
    - ✅ A **single input interface** for describing transformation situations
    - ✅ An **agentic analysis pipeline** with 8 specialised agents
    - ✅ **Live observability** (progress bar, metrics, "live agent thoughts")
    - ✅ An **analysis history** section for reviewing past runs
    """)

    st.markdown("### Scope of the Project")
    st.markdown("""
    This project focuses on **one primary use case** and one supporting use case:

    1. **Use Case A – Deep Agentic Transformation Analysis**  
       The user describes a transformation situation (e.g. project challenges, team dynamics,
       risks, and timelines). The assistant runs an 8-step agentic workflow to produce:
       - A structured analysis of risks and stakeholders  
       - Relevant change management frameworks  
       - A prioritised action plan  
       - An executive-style synthesis report  

    2. **Use Case B – Analysis History & Reflection**  
       After each run, the key input and final report are saved to a lightweight history.
       Users can revisit recent analyses to:
       - Review earlier situations and recommendations  
       - Compare multiple cases over time  
       - Reuse or refine insights for follow-up actions  
    """)

    st.markdown("### Objectives")
    st.markdown("""
    The objectives of this project are to:

    - **Demonstrate an agentic pattern** using multiple specialised AI agents instead of a single prompt.
    - **Support transformation decision-making** with structured, repeatable analysis.
    - **Show observability features** (time taken per agent, progress, live thoughts) so users can
      understand *how* the AI reaches its recommendations.
    - Implement **basic access control** (login) and safe usage patterns for LLMs.
    """)

    st.markdown("### Intended Users")
    st.markdown("""
    This assistant is designed for:

    - Transformation / change management teams  
    - Project and programme managers  
    - Digital transformation offices  
    - Leaders who need quick, structured insight from narrative updates
    """)

    st.markdown("### Data Sources & Privacy")
    st.markdown("""
    - The only primary data source is the **text that the user types into the app**.  
    - No external documents or databases are queried in this prototype.  
    - The app does **not** store data permanently in a database; recent runs are kept only in
      the in-memory **session state** for the current session's "Analysis History".  
    - The LLM backend (e.g. OpenAI GPT-4) is used for:
      - Understanding the user's description  
      - Running each specialised agent  
      - Generating the final report  

    This prototype is intended for demonstration and learning purposes only. Users should avoid
    entering confidential or personal data.
    """)

    st.markdown("### Key Features")
    st.markdown("""
    - 🔐 **Password-protected login** using hashed demo credentials  
    - 🤝 **8 specialised AI agents**:
        - 🎯 Orchestrator  
        - 📊 Data Extraction  
        - ⚠️ Risk Analysis  
        - 👥 Stakeholder Analysis  
        - 📚 Framework Advisor  
        - 📋 Action Planning  
        - ✓ Quality Assurance  
        - 🎨 Synthesis  
    - 📊 **Observability dashboard**:
        - Overall progress bar  
        - Time elapsed and estimated remaining time  
        - Per-agent status and timing  
        - "Live Agent Thoughts" section showing intermediate outputs  
    - 📜 **Analysis History** showing recent runs for quick review

    Together, these features show how agentic AI can support transformation management in a
    structured, explainable way.
    """)

# Main application logic
if not st.session_state.authenticated:
    login_page()
else:
    # Simple navigation inside the authenticated area
    with st.sidebar:
        st.markdown("### 📄 Pages")
        page = st.radio(
            "Go to",
            ["Agentic Assistant", "About Us", "Methodology"],
            index=0
        )

    if page == "Agentic Assistant":
        main_app()
    elif page == "About Us":
        about_page()
    elif page == "Methodology":
        methodology_page()
