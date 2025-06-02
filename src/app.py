import streamlit as st
from ai import call_ai
from langchain_core.messages import HumanMessage, AIMessage


st.set_page_config(page_title="ChatBot para Nfes", page_icon="🤖")
st.title("Ai Assistant - Os promptados")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá!! Estou aqui para te ajudar com as NFs"),
    ]
    
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)


user_input = st.chat_input("Digite sua pergunta aqui...")

if user_input is not None and user_input != "":
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    with st.chat_message("human"):
        st.markdown(user_input)
        
    with st.chat_message("ai"):
        response = st.write_stream(call_ai(user_input, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(content=response))


