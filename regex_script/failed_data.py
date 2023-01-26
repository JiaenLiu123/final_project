# Create a streamlit app that shows the data that failed to be processed

import streamlit as st
import pandas as pd
import numpy as np
import os


# Load the data
df = pd.read_csv("test.csv")


def color_coding(row):
    try:
        return (
            ["background-color:red"] * len(row)
            if int(row.new_date.split(", ")[1].split("(")[1]) < 2012
            else ["background-color:black"] * len(row)
        )
    except:
        return ["background-color:blue"] * len(row)


# Show the data
st.write(df.style.apply(color_coding, axis=1))
