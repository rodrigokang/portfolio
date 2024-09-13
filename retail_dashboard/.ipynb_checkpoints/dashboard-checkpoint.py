# <<<<<<<<<<<<<<<<<<<<<<<<<<< Description >>>>>>>>>>>>>>>>>>>>>>>>>>> #
#
# This Streamlit application displays a dashboard with two tabs: 
# Customers and Products. It fetches data from a Flask API, displays 
# the data in tables, and provides download options in CSV format. 
# The Products tab includes additional indicators showing the average 
# Units and Price. A login page is included for authentication, and a 
# logout button is provided.
#
# ******************************************************************* #

import streamlit as st
import requests
import pandas as pd
from pathlib import Path

# Define the base URL of your Flask application
base_url = 'http://127.0.0.1:5000'

def authenticate_user(username, password):
    """
    Authenticate user based on username and password.

    Args:
        username (str): Username for authentication.
        password (str): Password for authentication.

    Returns:
        bool: True if authentication is successful, False otherwise.
    """
    return username == "admin" and password == "password"

def login_page():
    """
    Display the login page for user authentication.
    """
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state.authenticated = True
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

def fetch_customers():
    """
    Fetch customer data from the API.

    Returns:
        pd.DataFrame: DataFrame containing customer data.
    """
    try:
        response = requests.get(f'{base_url}/api/customers')
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f'Failed to fetch customers data: {e}')
        return None

def fetch_products():
    """
    Fetch product data from the API.

    Returns:
        pd.DataFrame: DataFrame containing product data.
    """
    try:
        response = requests.get(f'{base_url}/api/products')
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f'Failed to fetch products data: {e}')
        return None

def convert_df_to_csv(df):
    """
    Convert a DataFrame to CSV format.

    Args:
        df (pd.DataFrame): DataFrame to be converted.

    Returns:
        bytes: CSV data in bytes format.
    """
    return df.to_csv(index=False).encode('utf-8')

def main():
    """
    Main function to run the Streamlit application.
    """
    if 'authenticated' not in st.session_state or not st.session_state.authenticated:
        login_page()
    else:
        st.title("Retail Dashboard")

        # Load CSS
        st.markdown('<link rel="stylesheet" href="/static/styles.css">', unsafe_allow_html=True)

        # Create a container for the main content
        main_container = st.container()
        logout_container = st.container()

        # Main content in the container
        with main_container:
            tab1, tab2 = st.tabs(["Customers", "Products"])
            
            with tab1:
                st.header("Customer Data")
                customers_df = fetch_customers()
                if customers_df is not None:
                    st.dataframe(customers_df)
                    csv = convert_df_to_csv(customers_df)
                    st.download_button(
                        label="Download Customer Data as CSV",
                        data=csv,
                        file_name='customers_data.csv',
                        mime='text/csv'
                    )
            
            with tab2:
                st.header("Product Data")
                products_df = fetch_products()
                if products_df is not None:
                    avg_units = products_df['Quantity'].mean()
                    avg_price = products_df['Price'].mean()
                    
                    # Load HTML for indicators
                    indicator_html = Path('static/indicators.html').read_text()
                    indicator_html = indicator_html.replace('{{ avg_units }}', f'{avg_units:.2f}')
                    indicator_html = indicator_html.replace('{{ avg_price }}', f'{avg_price:.2f}')
                    st.markdown(indicator_html, unsafe_allow_html=True)
                    
                    st.dataframe(products_df)
                    csv = convert_df_to_csv(products_df)
                    st.download_button(
                        label="Download Product Data as CSV",
                        data=csv,
                        file_name='products_data.csv',
                        mime='text/csv'
                    )

        # Add logout button to a separate container to position it
        with logout_container:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.write("")  # Add empty space to push the button to the bottom
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.experimental_rerun()

if __name__ == "__main__":
    main()