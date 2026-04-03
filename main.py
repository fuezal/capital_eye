import streamlit as st
st.title(" 📈 Capital Eye")
st.caption("Herramienta financiera con IA para inversionistas")

prompt = st.chat_input("¿En que te puedo ayudar?")
if prompt:
    st.write("El usuario ha enviado el siguiente mensaje: ", prompt)