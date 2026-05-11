import streamlit as st

st.title("🔐 Login")

# Simple demo credentials
USERNAME = "admin"
PASSWORD = "1234"

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):

    if username == USERNAME and password == PASSWORD:
        st.session_state["logged_in"] = True
        st.success("✅ Login Successful")
        st.switch_page("pages/2_Upload_Data.py")

    else:
        st.error("❌ Invalid Credentials")
