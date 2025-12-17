import streamlit as st
import requests
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="📋",
    layout="wide"
)

# Title
st.title("📋 SHL Assessment Recommendation System")
st.markdown("---")

# Description
st.markdown("""
This tool recommends relevant SHL assessments based on job descriptions or queries.
Enter a job description or natural language query below to get personalized assessment recommendations.
""")

# API URL
API_URL = "https://bhartendu1-shl-assessment.hf.space"

# Input Section
st.subheader("🔍 Enter Your Query")

query_type = st.radio(
    "Select input type:",
    ["Text Query", "Job Description URL"]
)

if query_type == "Text Query":
    query = st.text_area(
        "Enter job description or query:",
        placeholder="e.g., I am hiring for Java developers who can also collaborate effectively with my business teams.",
        height=150
    )
else:
    query = st.text_input(
        "Enter Job Description URL:",
        placeholder="https://example.com/job-description"
    )

# Submit Button
if st.button("🚀 Get Recommendations", type="primary"):
    if query:
        with st.spinner("Finding best assessments..."):
            try:
                # Call API
                response = requests.post(
                    f"{API_URL}/recommend",
                    json={"query": query},
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get("success") and data.get("recommendations"):
                        st.success(f"✅ Found {len(data['recommendations'])} recommendations!")
                        
                        st.subheader("📊 Recommended Assessments")
                        
                        # Create DataFrame for display
                        results = []
                        for i, rec in enumerate(data["recommendations"], 1):
                            results.append({
                                "Rank": i,
                                "Assessment Name": rec.get("name", "N/A"),
                                "URL": rec.get("url", "N/A"),
                                "Test Type": rec.get("test_type", "N/A"),
                                "Score": f"{rec.get('score', 0):.4f}"
                            })
                        
                        df = pd.DataFrame(results)
                        
                        # Display as table
                        st.dataframe(
                            df,
                            column_config={
                                "URL": st.column_config.LinkColumn("URL")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Also show as cards
                        st.subheader("📝 Detailed View")
                        for i, rec in enumerate(data["recommendations"], 1):
                            with st.expander(f"{i}. {rec.get('name', 'N/A')}"):
                                st.write(f"**URL:** {rec.get('url', 'N/A')}")
                                st.write(f"**Test Type:** {rec.get('test_type', 'N/A')}")
                                st.write(f"**Description:** {rec.get('description', 'N/A')}")
                                st.write(f"**Relevance Score:** {rec.get('score', 0):.4f}")
                    else:
                        st.warning("No recommendations found. Try a different query.")
                else:
                    st.error(f"API Error: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                st.error("Request timed out. Please try again.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a query or URL.")

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
**SHL Assessment Recommender**

This system uses AI to recommend relevant SHL assessments based on:
- Job descriptions
- Natural language queries
- Job posting URLs

**How it works:**
1. Enter your query
2. AI analyzes the requirements
3. Returns top 10 matching assessments

**Built with:**
- FastAPI
- Sentence Transformers
- FAISS Vector Search
- Streamlit
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**API Endpoint:**")
st.sidebar.code(API_URL)

# Footer
st.markdown("---")
st.markdown("Built for SHL Assessment Task | Powered by AI 🤖")
