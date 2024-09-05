# Agent class
### responsbility definition: expertise, scope, conversation script, style 
from openai import AzureOpenAI
import os
from pathlib import Path  
import json
import os 
from pprint import pprint
import requests
import time
# from langchain.utilities import BingSearchAPIWrapper
from datetime import datetime
import sys
from dotenv import load_dotenv
import inspect
import requests  
from concurrent.futures import ThreadPoolExecutor, as_completed 
from azure.search.documents.models import (

    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizedQuery,
)
from azure.search.documents import SearchClient  
from azure.core.credentials import AzureKeyCredential  
from PIL import Image
import io
import base64
import streamlit as st
from mimetypes import guess_type
import tempfile


env_path = Path('.') / 'secrets_sample.env'
load_dotenv(dotenv_path=env_path)
chat_engine =os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")


bing_subscription_key = os.environ['BING_SUBSCRIPTION_KEY']
bing_endpoint = os.environ['BING_SEARCH_URL'] 
client = AzureOpenAI(
  api_key=os.environ.get("AZURE_OPENAI_API_KEY"),  
  api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
  azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
)
emb_engine = os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT")
azure_search_client = SearchClient(  
    endpoint=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT"),  
    index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),  
    credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY"))  
)  

def bing_search(query):
    # Construct a request
    mkt = 'en-US'
    params = { 'q': query, 'mkt': mkt,"textDecorations": True, "textFormat": "HTML" }
    headers = { 'Ocp-Apim-Subscription-Key': bing_subscription_key }

    # Call the API
    response = requests.get(bing_endpoint, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()

    rows = "\n".join(["""<tr>
                        <td><a href=\"{0}\">{1}</a></td>
                        <td>{2}</td>
                        </tr>""".format(v["url"], v["name"], v["snippet"])
                    for v in search_results["webPages"]["value"]])
    search_results="<table>{0}</table>".format(rows)
    return "Here is the result from internet search:\n\n"+search_results
def get_embedding(text, model=emb_engine):  
    text = text.replace("\n", " ")  
    embedding_response = client.embeddings.create(input = [text], model=model).data[0].embedding
    return embedding_response  

def local_search(search_query):

    vector = VectorizedQuery(vector=get_embedding(search_query), k_nearest_neighbors=3, fields="summaryVector")

    results = azure_search_client.search(  
        search_text=search_query,  
        vector_queries= [vector],
        query_type=QueryType.SEMANTIC, semantic_configuration_name='my-semantic-config', query_caption=QueryCaptionType.EXTRACTIVE, query_answer=QueryAnswerType.EXTRACTIVE,
        select=["summary","content_details"],
        top=3
    )  
    text_content ="Here is the result of internal search tool: \n\n"
    for result in results:  
        text_content += f"{result['summary']}\n{result['content_details']}\n\n"
    # print("text_content", text_content)
    return text_content
def search(query):  
    # Dictionary to store the results  
    results = {}  
      
    # Perform searches concurrently  
    with ThreadPoolExecutor() as executor:  
        future_to_search = {  
            executor.submit(local_search, query): 'local_search',  
            executor.submit(bing_search, query): 'bing_search'  
        }  
          
        for future in as_completed(future_to_search):  
            search_type = future_to_search[future]  
            try:  
                result = future.result()  
                results[search_type] = result  
            except Exception as exc:  
                results[search_type] = f'{search_type} generated an exception: {exc}'  
      
    # Combine results, ensuring local search result comes first  
    ordered_results = []  
    if 'local_search' in results:  
        ordered_results.append(results['local_search'])  
    if 'bing_search' in results:  
        ordered_results.append(results['bing_search'])  
      
    return "\n\n".join(ordered_results)    


# get the curent date and time in the format: YYYY-MM-DD 
def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")
PERSONA = f"""
You are a friendly and knowledgeable local AI guide for Madame Lân Restaurant in Da Nang, Vietnam. 
The current date/time is {get_current_date()}. 
Your mission is to help users with any general knowledge questions they have about Madame Lân restaurant. 
If the information requires up-to-date knowledge, use a search tool to provide accurate and helpful answers. 
Be thorough in your research by performing multiple searches if necessary. 
If you cannot find the answer, be honest and inform the user. 
The search tool is a combination of both local database search and internet search. 
If user asks for nutrition facts, use your based knowledge to answer it.
Always highly prioritize local search results if both return usable results. 
Always respond in the same language as the user. Ensure that your answers are brief and concise. 
"""

def check_args(function, args):
    sig = inspect.signature(function)
    params = sig.parameters

    # Check if there are extra arguments
    for name in args:
        if name not in params:
            return False
    # Check if the required arguments are provided 
    for name, param in params.items():
        if param.default is param.empty and name not in args:
            return False

    return True



AVAILABLE_FUNCTIONS = {
            "search": search,

        } 

# "A search tool that combines result from both internal data source and internet search"

FUNCTIONS_SPEC= [ 

    {
        
        "type":"function",
        "function":{

        "name": "search",
        "description": "A search tool that combines result from both internal data source and internet search",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"],
        }},
    },
]  

# def save_uploadedfile(uploadedfile):
#     path_to_dir = f"tempDir/{str(uploadedfile.name)}"
#     with open(path_to_dir,"wb") as f:
#         f.write(uploadedfile.getvalue())
#     return st.success("Saved File:{} to tempDir".format(str(uploadedfile.name)))


def local_image_to_data_url(image):
    base64_encoded_data = base64.b64encode(image.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{image.type};base64,{base64_encoded_data}"


class Smart_Agent():
    """
    Agent that can use other agents and tools to answer questions.

    Args:
        persona (str): The persona of the agent.
        tools (list): A list of {"tool_name":tool} that the agent can use to answer questions. Tool must have a run method that takes a question and returns an answer.
        stop (list): A list of strings that the agent will use to stop the conversation.
        init_message (str): The initial message of the agent. Defaults to None.
        engine (str): The name of the GPT engine to use. Defaults to "gpt-35-turbo".

    Methods:
        llm(new_input, stop, history=None, stream=False): Generates a response to the input using the LLM model.
        _run(new_input, stop, history=None, stream=False): Runs the agent and generates a response to the input.
        run(new_input, history=None, stream=False): Runs the agent and generates a response to the input.

    Attributes:
        persona (str): The persona of the agent.
        tools (list): A list of {"tool_name":tool} that the agent can use to answer questions. Tool must have a run method that takes a question and returns an answer.
        stop (list): A list of strings that the agent will use to stop the conversation.
        init_message (str): The initial message of the agent.
        engine (str): The name of the GPT engine to use.
    """


    def __init__(self, persona,functions_spec, functions_list, name=None, init_message=None, engine =chat_engine):
        if init_message is not None:
            init_hist =[{"role":"system", "content":persona}, {"role":"assistant", "content":init_message}]
        else:
            init_hist =[{"role":"system", "content":persona}]

        self.init_history =  init_hist
        self.persona = persona
        self.engine = engine
        self.name= name

        self.functions_spec = functions_spec
        self.functions_list= functions_list
        
    # @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def run(self, user_input, conversation=None, image=None):
        if user_input is None: #if no input return init message
            return self.init_history, self.init_history[1]["content"]
        if conversation is None: #if no history return init message
            conversation = self.init_history.copy()
        

        if image is not None:
            data_url = local_image_to_data_url(image)
            conversation.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_input
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ]
            })
        else:
            conversation.append({"role": "user", "content": user_input})

        request_help = False
        while True:
            response = client.chat.completions.create(
                model=self.engine, # The deployment name you chose when you deployed the GPT-35-turbo or GPT-4 model.
                messages=conversation,
            tools=self.functions_spec,
            tool_choice='auto',
            max_tokens=600,

            )
            
            response_message = response.choices[0].message
            if response_message.content is None:
                response_message.content = ""

            tool_calls = response_message.tool_calls
            

            print("assistant response: ", response_message.content)
            # Step 2: check if GPT wanted to call a function
            if  tool_calls:
                conversation.append(response_message)  # extend conversation with assistant's reply
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    print("Recommended Function call:")
                    print(function_name)
                    print()
                
                    # Step 3: call the function
                    # Note: the JSON response may not always be valid; be sure to handle errors
                                    
                    # verify function exists
                    if function_name not in self.functions_list:
                        # raise Exception("Function " + function_name + " does not exist")
                        conversation.pop()
                        continue
                    function_to_call = self.functions_list[function_name]
                    
                    # verify function has correct number of arguments
                    function_args = json.loads(tool_call.function.arguments)

                    if check_args(function_to_call, function_args) is False:
                        # raise Exception("Invalid number of arguments for function: " + function_name)
                        conversation.pop()
                        continue

                    
                    # print("beginning function call")
                    function_response = str(function_to_call(**function_args))

                    print("Output of function call:")
                    print(function_response)
                    print()
                
                    conversation.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        }
                    )  # extend conversation with function response
                    

                continue
            else:
                break #if no function call break out of loop as this indicates that the agent finished the research and is ready to respond to the user

        conversation.append(response_message)
        assistant_response = response_message.content

        return request_help, conversation, assistant_response
