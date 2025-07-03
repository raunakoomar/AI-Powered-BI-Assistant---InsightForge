import streamlit as st
import pandas as pd
from analysis.summary import load_data, generate_summary
from utils.visualizer import (
    plot_monthly_sales,
    plot_region_comparison,
    plot_product_performance,
    plot_customer_demographics
)
from llm.rag_engine import InsightForgeRAG

st.set_page_config(page_title="InsightForge", layout="wide")
st.title("📊 InsightForge: AI-Powered Business Intelligence Assistant")

# Load and summarize business data
df = load_data()
summary = generate_summary(df)

# Build the RAG system
rag = InsightForgeRAG(summary_text=str(summary), df=df)

# User input
query = st.text_input("Ask a business question (e.g. top product, sales trends, regional performance):")

if st.button("Submit") and query:
    st.subheader("🧠 Insight")
    response = rag.answer(query)
    st.write(response)

st.markdown("---")
st.subheader("📈 Data Visualizations")

col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_monthly_sales(df))
    st.pyplot(plot_product_performance(df))
with col2:
    st.pyplot(plot_region_comparison(df))
    st.pyplot(plot_customer_demographics(df))