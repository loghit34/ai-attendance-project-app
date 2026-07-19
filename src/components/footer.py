import streamlit as st


def footer_home():
    st.markdown(f"""
        <div style="margin-top:2rem; display:flex; gap:6px; justify-content:center; align-items:center;">
        <p style="font-weight:bold; color:white; margin:0;">Created with &nbsp; ❤️ &nbsp; by &nbsp; <span style="color:white;">Lohit Ghosh</span></p>  
        </div>
                
                """, unsafe_allow_html=True)


def footer_dashboard():
    st.markdown(f"""
        <div style="margin-top:2rem; display:flex; gap:6px; justify-content:center; align-items:center;">
        <p style="font-weight:bold; color:black; margin:0;">Created with &nbsp; ❤️ &nbsp; by &nbsp; <span style="color:black;">Lohit Ghosh</span></p>  
        </div>
                
                """, unsafe_allow_html=True)
