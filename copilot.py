import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from copilot_utils import Smart_Agent, PERSONA, AVAILABLE_FUNCTIONS, FUNCTIONS_SPEC
import sys
import time
import random
import os
from pathlib import Path  
import json
agent = Smart_Agent(persona=PERSONA,functions_list=AVAILABLE_FUNCTIONS, functions_spec=FUNCTIONS_SPEC, init_message="Hi there, this is Mỹ Hương, your AI guide in Madame Lân restaurant. How can I help you?")

st.set_page_config(layout="wide",page_title="Madame Lân AI Chatbot")
styl = f"""
<style>
    .stTextInput {{
      position: fixed;
      bottom: 3rem;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)


MAX_HIST= 3
# Sidebar contents
with st.sidebar:
    st.title('Madame Lân Restaurant')
    st.markdown('''
    A Demo of AI Assistant
    ''')
    add_vertical_space(5)
    st.write('@ Copyright AiAiVN 2024')
    if st.button('Clear Chat'):

        if 'history' in st.session_state:
            st.session_state['history'] = []

    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'input' not in st.session_state:
        st.session_state['input'] = ""
    if 'question_count' not in st.session_state:
        st.session_state['question_count'] = 0


user_input= st.chat_input("You:")
image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)


# picture = st.camera_input("Take a picture")

## Conditional display of AI generated responses as a function of user provided prompts
history = st.session_state['history']
question_count=st.session_state['question_count']

if len(history) > 0:
    idx=0
    removal_indices =[]
    running_question_count=0
    start_counting=False # flag to start including history items in the removal_indices list
    running_question_count=0
    start_counting=False # flag to start including history items in the removal_indices list
    for message in history:
        idx += 1
        message = dict(message)
        print("role: ", message.get("role"), "name: ", message.get("name"))
        if message.get("role") == "user":
            running_question_count +=1
            start_counting=True
        if start_counting and (question_count- running_question_count>= MAX_HIST):
            removal_indices.append(idx-1)
        elif question_count- running_question_count< MAX_HIST:
            break
            
    # remove items with indices in removal_indices
    # print("removal_indices", removal_indices)
    for index in removal_indices:
        del history[index]
    question_count=0
    # print("done purging history, len history now", len(history ))

    for message in history:
        idx += 1
        message = dict(message)
        if message.get("role") != "system" and message.get("role") != "tool" and message.get("name") is None and len(message.get("content")) > 0:
            with st.chat_message(message["role"]):
                    if type(message["content"]) == type([]):
                        st.markdown(message["content"][0]["text"]) 
                    else:
                        st.markdown(message["content"])
else:
    history, agent_response = agent.run(user_input=None)
    with st.chat_message("assistant"):
        st.markdown(agent_response)
    user_history=[]
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    query_used, history, agent_response = agent.run(user_input=user_input, conversation=history, image=image)
    with st.chat_message("assistant"):
        st.markdown(agent_response)

st.session_state['history'] = history