import streamlit as st
import mysql.connector

# Database connection
@st.cache_resource
def init_connection():
    return mysql.connector.connect(
        host=st.secrets["localhost"],
        user=st.secrets["root"],
        password=st.secrets["password"],
        database=st.secrets["customerreviews"]
    )

conn = init_connection()

# Perform query
@st.cache_data(ttl=600)
def run_query(query):
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    return result

# Function to insert data into the database
def insert_data(title, reviews, review_date, place):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO customerreviews (title, reviews, reviewDate, place) VALUES (%s, %s, %s, %s)",
        (title, reviews, review_date, place)
    )
    conn.commit()
    cursor.close()

# Input form
st.title("Customer Reviews Input Form")

with st.form("input_form"):
    title = st.text_input("Title")
    reviews = st.text_area("Reviews")
    review_date = st.date_input("Review Date")
    place = st.text_input("Place")
    submitted = st.form_submit_button("Submit")

    if submitted:
        insert_data(title, reviews, review_date, place)
        st.success("Review submitted successfully!")

# Display existing reviews
st.title("Existing Reviews")
rows = run_query("SELECT * FROM customerreviews")

for row in rows:
    st.write(f"SN: {row[0]}, Title: {row[1]}, Reviews: {row[2]}, Review Date: {row[3]}, Place: {row[4]}")
