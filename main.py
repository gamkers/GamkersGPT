import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq
import os

from io import StringIO, BytesIO



# Initialize ChatGroq with API key from environment
llm = ChatGroq(model_name='llama3-70b-8192', api_key='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')

def read_file(uploaded_file):
    """
    Read the uploaded file based on its type (Excel or CSV).
    """
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a CSV or Excel file.")
        return None

def process_data(df,query ):
    """
    Process the DataFrame using SmartDataframe and ChatGroq.

    """
    data = df
    smart_df = SmartDataframe(data, config={'llm': llm})
    # Chat query to generate scatter plot
    return smart_df.chat(query)


def main():
    st.title('Excel and CSV Data Processing with LLM - By Akash M')
    
    uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Read the file based on its type
        df = read_file(uploaded_file)
        
        if df is not None:
            # Display the DataFrame
            st.write("### Data Preview", df.head())

            query = st.text_input("Enter your question (e.g., 'Show a pie chart for the composition of a column')")

            if query:
                # Process the data and generate response
                 result = process_data(df,query)

            if st.button('Submit'):
                if query:
                    # Process the data and generate response
                    result = process_data(df, query)
                    
                    st.write("### Data Output")
                    
                    if str(type(result)) == "<class 'int'>":
                        st.write(result)
                    elif str(type(result)) == "<class 'str'>":
                        st.write(result)
                    elif str(type(result)) == "<class 'numpy.int64'>":
                        st.write(result)
                    else:
                        st.image(result, use_column_width=True)
                else:
                    st.warning("Please enter a query before submitting.") 

            # Display the result (scatter plot)
            # st.write("### Scatter Plot")
            
if __name__ == "__main__":
    main()
