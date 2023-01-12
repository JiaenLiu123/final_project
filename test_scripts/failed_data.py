# Create a streamlit app that shows the data that failed to be processed

import streamlit as st
import pandas as pd
import numpy as np
import os


# Load the data
df = pd.read_csv("/Users/liujiaen/Documents/Text_Recognition/final_project/test_scripts/test.csv")

# Show the data
st.write(df)