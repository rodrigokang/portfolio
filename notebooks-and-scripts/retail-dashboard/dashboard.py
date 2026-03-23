# Import libraries
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Define the base URL of Flask application
base_url = 'https://myapi-segmentation-model-dashboard.onrender.com'

def authenticate_user(username, password):
    """
    Authenticate a user based on provided username and password.

    Args:
        username (str): The username input from the user.
        password (str): The password input from the user.

    Returns:
        bool: True if authentication is successful, otherwise False.
    """
    return username == "admin" and password == "password"

def login_page():
    """
    Display a login page to authenticate users.
    
    This function renders a simple login form where users can enter
    their username and password. If the credentials are correct,
    the user is authenticated and granted access to the application.
    """
    st.title("Login Page")

    # Display a banner with login information
    st.info("For testing purposes, use the following credentials: \n\n**Username**: admin\n**Password**: password")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state.authenticated = True
        else:
            st.error("Invalid username or password")


def fetch_products():
    """
    Fetch product data from the Flask API.

    This function sends a GET request to the Flask API to retrieve
    the product data and returns it as a Pandas DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the product data, or None if
        an error occurs.
    """
    try:
        response = requests.get(f'{base_url}/api/products')
        response.raise_for_status()  # Check for HTTP errors
        data = response.json()  # Convert response to JSON
        df = pd.DataFrame(data)  # Convert JSON to DataFrame
        return df
    except requests.exceptions.RequestException as e:
        st.error(f'Failed to fetch products data: {e}')
        return None

def fetch_segmented_data(data):
    """
    Fetch segmented customer data from the Flask API based on the provided data.

    Args:
        data (dict): The data to be sent to the API, formatted as a dictionary.

    Returns:
        pd.DataFrame: DataFrame containing the segmented data, or None if
        an error occurs.
    """
    try:
        response = requests.post(f'{base_url}/api/segment', json=data)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f'Failed to fetch segmented data: {e}')
        return None

def convert_df_to_csv(df):
    """
    Convert a Pandas DataFrame to CSV format.

    Args:
        df (pd.DataFrame): The DataFrame to be converted.

    Returns:
        bytes: CSV data encoded in UTF-8.
    """
    return df.to_csv(index=False).encode('utf-8')

def create_donut_chart_city(df):
    """
    Create and display a donut chart showing the percentage of products by city.

    Args:
        df (pd.DataFrame): DataFrame containing product data with a 'City' column.

    Displays:
        A donut chart visualizing the percentage distribution of products by city.
    """
    city_counts = df['City'].value_counts(normalize=True) * 100
    fig, ax = plt.subplots(figsize=(8, 6))  # Smaller figure size
    wedges, texts, autotexts = ax.pie(city_counts, labels=city_counts.index, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    
    # Draw a circle at the center of pie to make it look like a donut
    center_circle = plt.Circle((0,0), 0.70, fc='white')
    fig.gca().add_artist(center_circle)

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')  
    plt.title("Percentage of Products by City")
    st.pyplot(fig)

def create_donut_chart_category(df):
    """
    Create and display a donut chart showing the total quantity of products by category.

    Args:
        df (pd.DataFrame): DataFrame containing product data with a 'CategoryName' column and 'Quantity' column.

    Displays:
        A donut chart visualizing the total quantity of products by category.
    """
    category_quantity = df.groupby('CategoryName')['Quantity'].sum()
    fig, ax = plt.subplots(figsize=(8, 6))  # Smaller figure size
    wedges, texts, autotexts = ax.pie(category_quantity, labels=category_quantity.index, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    
    # Draw a circle at the center of pie to make it look like a donut
    center_circle = plt.Circle((0,0), 0.70, fc='white')
    fig.gca().add_artist(center_circle)

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    plt.title("Total Quantity of Products by Category")
    st.pyplot(fig)

def create_barplot(df):
    """
    Create and display a bar plot showing the total quantity of products over time.

    Args:
        df (pd.DataFrame): DataFrame containing product data with a 'OrderDate' column and 'Quantity' column.

    Displays:
        A bar plot showing the total quantity of products grouped by date.
    """
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])  # Ensure 'OrderDate' is in datetime format
    products_by_date = df.groupby(df['OrderDate'].dt.date)['Quantity'].sum()  # Group by day

    plt.figure(figsize=(12, 3))  # Smaller figure size
    sns.barplot(x=products_by_date.index, y=products_by_date.values)
    plt.title("Total Quantity of Products Over Time")
    plt.xlabel("Order Date")
    plt.ylabel("Total Quantity")
    plt.xticks(rotation=45)  # Rotate the x-axis labels
    st.pyplot(plt)

def main():
    """
    Main function to run the Streamlit application.

    Handles user authentication, navigation between different pages,
    and data fetching, visualization, and download functionalities.
    """
    if 'authenticated' not in st.session_state or not st.session_state.authenticated:
        login_page()
    else:
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Select Page", ["Products", "Segmentation Model"])

        if page == "Products":
            st.title("Products Dashboard")

            # Fetch product data
            products_df = fetch_products()

            if products_df is not None:
                # Country filter
                country_selected = st.sidebar.selectbox("Select a Country", products_df['Country'].unique())

                # Filter products by selected country
                filtered_products_df = products_df[products_df['Country'] == country_selected]

                total_quantity = int(filtered_products_df['Quantity'].sum())
                avg_price = filtered_products_df['Price'].mean()
                total_distinct_customers = filtered_products_df['City'].nunique()  # Count distinct cities

                # Load HTML for the product indicators
                indicator_html_path = Path('static/indicators.html')
                if indicator_html_path.is_file():
                    indicator_html = indicator_html_path.read_text()
                    indicator_html = indicator_html.replace('{{ total_quantity }}', f'{total_quantity}')
                    indicator_html = indicator_html.replace('{{ avg_price }}', f'${avg_price:.2f}')
                    indicator_html = indicator_html.replace('{{ total_distinct_customers }}', 
                                                            f'{total_distinct_customers}')  # Update distinct customers
                    st.markdown(indicator_html, unsafe_allow_html=True)
                else:
                    st.error(f'HTML file not found at {indicator_html_path}')

                # Display the plots side by side
                col1, col2 = st.columns(2)

                with col1:
                    create_donut_chart_city(filtered_products_df)

                with col2:
                    create_donut_chart_category(filtered_products_df)

                create_barplot(filtered_products_df)

                st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                st.dataframe(filtered_products_df, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                csv = cont_df_to_csv(filtered_products_df)
                st.download_buvertton(
                    label="Download Product Data as CSV",
                    data=csv,
                    file_name='products_data.csv',
                    mime='text/csv'
                )

        elif page == "Segmentation Model":
            st.title("Segmentation Model")

            # Fetch product data without country filter
            products_df = fetch_products()
            if products_df is not None:
                if st.button("Run Segmentation"):
                    # Prepare the data to send to the backend
                    filtered_products_df = products_df.copy()
                    for col in filtered_products_df.columns:
                        if pd.api.types.is_datetime64_any_dtype(filtered_products_df[col]):
                            filtered_products_df[col] = filtered_products_df[col].dt.strftime('%Y-%m-%d')

                    data_to_send = {
                        'data': filtered_products_df.to_dict(orient='records')  # Convert DataFrame to dict
                    }

                    segmented_df = fetch_segmented_data(data_to_send)
                    if segmented_df is not None and not segmented_df.empty:
                        st.subheader("Segmentation Results")
                        st.dataframe(segmented_df, use_container_width=True)

                        # Convert segmented DataFrame to CSV and add download button
                        csv = convert_df_to_csv(segmented_df)
                        st.download_button(
                            label="Download Segmentation Data as CSV",
                            data=csv,
                            file_name='segmentation_data.csv',
                            mime='text/csv'
                        )
                    else:
                        st.error("No segmented data available or an error occurred.")

        # Sidebar logout button
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False

if __name__ == "__main__":
    main()