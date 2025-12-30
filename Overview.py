import streamlit as st

st.set_page_config(page_title="Scooter Demo", layout="wide")

st.header("Hello 👋")
st.markdown("""
Welcome to the Scooter Demo case study!  
Use the sidebar to navigate between the following pages:

- **Overview**: You are already here!
- **Data Exploration**: Explore the scooter dataset, visualize key features, and understand data distributions (and the many errors).
- **Model Training, Optimization & Evaluation**: Get an overview over the model, its results and evaluate your own dataset.

The full code can be found on [GitHub](https://github.com/PetrusBellmonte/scooter-demo).
""")

