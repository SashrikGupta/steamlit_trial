import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt 
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import GoogleGenerativeAI
import json

load_dotenv()

client = pymongo.MongoClient(os.getenv("DB_key"))
db = client["mydatabase"]
collection = db["users"]

llm  = GoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_KEY"))

def ask(query, agent):
    if "graph" in query:
        data_str = agent.invoke(query+"all x and all y values for making garaph and should be in format { 'x' :[all values of x] , 'y':[all values of y] } do not try to make graph just provide values")["output"]
        
        # Remove spaces around the keys to make it a valid JSON
        data_str = data_str.replace("'", "\"")

        # Parse the JSON string
        data_dict = json.loads(data_str)

        # Extract x and y values
        x_values = data_dict['x']
        y_values = data_dict['y']
        
        # Plot the graph using Streamlit
        chart_type = st.selectbox('Select Chart Type', ['line', 'bar', 'pie'])
        if chart_type == 'line':
            st.line_chart(pd.DataFrame({'x': x_values, 'y': y_values}))
        elif chart_type == 'bar':
            st.bar_chart(pd.DataFrame({'x': x_values, 'y': y_values}))
        elif chart_type == 'pie':
            st.pie_chart(pd.DataFrame({'x': x_values, 'y': y_values}))

    else:
        result = agent.invoke(query)["output"]
        st.write(result)
        return result

def main():

    st.write("# CHAT WITH YOUR CSV DATA SET")
    uploaded_file = st.file_uploader('Choose a csv file', type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write('Uploaded Dataset:')
        st.write(df)

        user_text = st.text_input('Query regarding the dataset')

        agent = create_pandas_dataframe_agent(
            llm,
            df,
            allow_dangerous_code=True,
            verbose=True,
        )

        if user_text:
            st.write("## AI MODEL")
            ask(user_text, agent)


if __name__ == '__main__':
    main()
