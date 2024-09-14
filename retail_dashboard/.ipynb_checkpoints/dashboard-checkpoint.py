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
        else:
            st.error("Invalid username or password")

def fetch_products():
    """
    Fetch product data from the API.
    
    Returns:
        pd.DataFrame: DataFrame containing product data, or None if an error occurs.
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

        # Sidebar with country selection
        st.sidebar.title("Filter")
        products_df = fetch_products()

        if products_df is not None:
            countries = products_df['Country'].unique().tolist()
            countries.append("All countries")  # Add option to select all countries
            country_selected = st.sidebar.selectbox("Select a Country", countries)

            # Sidebar logout button
            if st.sidebar.button("Logout"):
                st.session_state.authenticated = False

            # Filter products by selected country
            if country_selected == "All countries":
                filtered_products_df = products_df
                total_distinct_customers = products_df['CustomerID'].nunique()
            else:
                filtered_products_df = products_df[products_df['Country'] == country_selected]
                total_distinct_customers = products_df[products_df['Country'] == country_selected]['CustomerID'].nunique()

            total_quantity = filtered_products_df['Quantity'].sum()
            avg_price = filtered_products_df['Price'].mean()

            # Display Sales Analysis title
            st.markdown("<h2>Sales Analysis</h2>", unsafe_allow_html=True)

            # Load HTML for the product indicators
            indicator_html_path = Path('static/indicators.html')
            if indicator_html_path.is_file():
                indicator_html = indicator_html_path.read_text()
                indicator_html = indicator_html.replace('{{ total_distinct_customers }}', f'{total_distinct_customers}')
                indicator_html = indicator_html.replace('{{ total_quantity }}', f'{total_quantity}')
                indicator_html = indicator_html.replace('{{ avg_price }}', f'${avg_price:.2f}')
                st.markdown(indicator_html, unsafe_allow_html=True)
            else:
                st.error(f'HTML file not found at {indicator_html_path}')

            # Add a container to make the dataframe responsive
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(filtered_products_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Download the filtered products CSV
            csv = convert_df_to_csv(filtered_products_df)
            st.download_button(
                label="Download Product Data as CSV",
                data=csv,
                file_name='products_data.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    main()
