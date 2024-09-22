from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from pydantic import BaseModel, Field
# from langchain.utilities.wikipedia import WikipediaAPIWrapper # change to WikipediaRetriever
from langchain.retrievers import WikipediaRetriever
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
import json
import openai as client
import streamlit as st
from typing import Type
from bs4 import BeautifulSoup
import requests
import time
# 새로운 주제로 넘기는 버튼
# 이어서 대화 가능하도록 아이디 그래도 유지하면서 새로운 주제로 넘기기
st.set_page_config(
    page_title="OpenAI_Assistants",
    page_icon="📃",
)

if 'messages' not in st.session_state:
    st.session_state["messages"] = []
if 'count' not in st.session_state:
    st.session_state['count'] = 0
if 'download_buttons' not in st.session_state:
    st.session_state['download_buttons'] = []


def DuckDuckSearchTool(inputs):
    print("DuckDuck search tool inputs:",inputs)
    ddg = DuckDuckGoSearchAPIWrapper()
    print("DuckDuck search start")
    query = inputs["query"]
    results = ddg.run(query)
    send_message(f"This is the DuckDuckSearchTool brief result \n\n DuckDuckSearchTool result: {results[:500]}.....", "assistant")  # Send only text content to the user
    # # Assuming the results are a JSON object and the first result's URL is accessible via ['results'][0]['url']
    # if results and 'results' in results and len(results['results']) > 0:
    #     url = results[0]['link']  # Access the URL of the first result
    #     print("DuckDuck url:",url)
    print("DuckDuck results:",results)    
    return results


# def WikiSearchTool(inputs):
#     print("Wiki search tool inputs:",inputs)
#     wiki = WikipediaAPIWrapper()
#     query = inputs["query"]
#     results = wiki.run(query)
#     print("Wiki results:",results)    
#     return results

def WikiSearchTool(inputs):
    print("Wiki search tool inputs:",inputs)
    retriever = WikipediaRetriever(top_k_results=3)
    query = inputs["query"]
    docs = retriever.invoke(query)
    print("Wiki results:",docs)
    results = ""
    results_display = ""
    for content in docs:
        results += f"{content.page_content}\n"
        results_display += f"Title: {content.metadata.get('title', 'No title')}\n\n"
        results_display += f"Summary: {content.page_content[:300]}....."  # Limit content to the first 500 characters for brevity
        results_display += "\n\n"        
    send_message(f"This is the WikiSearchTool brief result \n\n WikiSearchTool result: {results_display}", "assistant")  # Send only text content to the user
    return results

def SaveToTextFileTool(inputs):
    try:
        file_path = f"MyResearch.txt"
        text = inputs["text"]

        if 'count' not in st.session_state:
            st.session_state['count'] = 0
        mykey = st.session_state['count']
        # if "url" in inputs.keys():
        #     url = inputs["url"]
        #     response = requests.get(url)
        #     soup = BeautifulSoup(response.content, "html.parser")
        #     response.raise_for_status()  # Raises an error for bad responses
        #     text = soup.get_text()
        #     st.download_button('Download text', text, 'text',key = mykey)
        #     # with open(file_path, 'w') as f:
        #     #     f.write(text) 
        #     #     # st.download_button('Download CSV', f) 
        #     #     print(f"Saved text from URL to file: {file_path}")
        # else:
        download_id = f"download_{mykey}"

        download_button_data = {
            'id': download_id,
            'text': text,
            'label': 'Download text',
            'file_name': file_path
        }

        st.session_state['count'] += 1

        # Send message with download_button_data
        send_message(
            f"This is the result \n result: {text}",
            "assistant",
            download_button_data=download_button_data
        )
        # with open(file_path, 'w') as f:
        #     f.write(text)
        #     print(f"Saved text from URL to file: {file_path}")
        st.session_state['count'] += 1

        return "Text saved to file."
    except Exception as e:
        return f"An error occurred: {str(e)}"
    

def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        try:
            print(json.loads(function.arguments))
            outputs.append(
                {
                    "output": functions_map[function.name](json.loads(function.arguments)),
                    "tool_call_id": action_id,
                }
            )
        except Exception as e:
            print(f"Error calling{function.arguments} : {str(e)}")
            outputs.append(
                {
                    "output": f"Error calling function {function.name}: {str(e)}",
                    "tool_call_id": action_id,
                }
            )
    return outputs    

def submit_tool_outputs(run_id, thread_id):
    outpus = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id, thread_id=thread_id, tool_outputs=outpus
    )
@st.cache_data
def create_run(thread_id, assistant_id):
    # Retrieve the list of runs associated with the thread
    # 실행중 오류 발생 후 해당 키워드 다시 검색 시 실행중인 run이 있으면 취소하고 새로운 run 생성
    runs = client.beta.threads.runs.list(thread_id=thread_id)
    active_run = None
    for run in runs:
        if run.status in ['queued', 'in_progress', 'requires_action']:
            active_run = run
            break

    if active_run:
        print(f"Active run {active_run.id} found for thread {thread_id}. Waiting for it to complete or canceling it.")
        # Optionally, you can wait for it to complete or cancel it
        # Here, we'll cancel the active run
        client.beta.threads.runs.cancel(run_id=active_run.id, thread_id=thread_id)
        # Wait for the run to be canceled
        while True:
            run_status = client.beta.threads.runs.retrieve(run_id=active_run.id, thread_id=thread_id).status
            if run_status == 'cancelled':
                break
            time.sleep(1)

    # Now, create a new run
    return client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )
@st.cache_data
def create_thread(content):
    return client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ]
    )
     
functions = [
    {
        "type": "function",
        "function": {
            "name": "DuckDuckSearchTool",
            "description": "Use this tool to search for the provided query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query that user wants to search",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "WikiSearchTool",
            "description": "Use this tool to search for the provided query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query that user wants to search",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "SaveToTextFileTool",
            "description": "Use this tool to save the text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text that user wants to save to the text file.",
                    },
                },
                "required": ["text"],
            },
        },
    },
]



        
    
functions_map = {
    "DuckDuckSearchTool": DuckDuckSearchTool,
    "WikiSearchTool": WikiSearchTool,
    "SaveToTextFileTool": SaveToTextFileTool,
}



count = 0

# Sidebar for API Key and GitHub Repo link
with st.sidebar:
    docs = None
    topic = None
    api_key = st.text_input("Enter your OpenAI API Key")
    # todo: if user change the api key, we need to reinitialize the llm
    # how to save api key in a secure way?
    API_key_check_btn = st.button("Check API KEY")
    if not api_key:
        st.error("Please enter your OpenAI API Key")    
    if st.session_state.get("APIKEY_failed"):  
        st.error("Invalid API Key. Please enter a valid API Key")
    chat_model = st.selectbox("Select GPT model", ["gpt-4o-mini", "gpt-4o","gpt-4-turbo","gpt-3.5-turbo","o1-preview","o1-mini"])  #"o1-preview","o1-mini" not support yet
# Check if the selected model is unsupported
    if chat_model in ["o1-preview", "o1-mini"]:
        st.warning(f"Model '{chat_model}' is not supported. We will update as soon as they support 😉.\n * Automatically switching to 'gpt-4o-mini'.")
        chat_model = "gpt-4o-mini"
     
    st.write("GitHub Repo:https://github.com/daegyujeong/gpt_practice_dk_v2/blob/f66ccd2ecb0ecd7d0673febf9509b9073b66395f/pages/05_OpenAI_Assistants.py")
    if API_key_check_btn:
        if  api_key:
            llm = None
            # for checking if the API Key is valid
            try:
                llm = ChatOpenAI(
                    api_key=api_key,
                    temperature=0.1,
                    model="gpt-3.5-turbo-1106",
                )
                llm.invoke("Hello!")   
                st.session_state["APIKEY_failed"] = False
                st.success("API Key is valid")
                # You might want to add a simple test call here if the API does not get invoked elsewhere immediately
            except Exception as e:
                st.markdown(f"Failed to initialize ChatOpenAI: {str(e)}")   
                st.session_state["APIKEY_failed"] = True
                st.rerun()      
 


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(
        run_id=run_id,
        thread_id=thread_id,
    )

def get_required_action(run_id, thread_id):
    run = get_run(run_id, thread_id)
    return run.required_action

def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    messages = list(messages)
    last_message = {"role":"", "content":""}
    if len(messages) > 0:
        last_message = {"role":messages[0].role, "content":messages[0].content[0].text.value}
    messages.reverse()
    for message in messages:
        print(f"{message.role}: {message.content[0].text.value}")
        # for debugging
    return last_message
    
def save_message(message, role, download_id=None):
    st.session_state["messages"].append({
        "message": message,
        "role": role,
        "download_id": download_id
    })

def send_message(message, role, save=True, download_button_data=None):
    with st.chat_message(role):
        st.markdown(message)
        if download_button_data:
            # Initialize the 'download_buttons' list if it doesn't exist
            if 'download_buttons' not in st.session_state:
                st.session_state['download_buttons'] = []

            # Store the download button data in session state
            st.session_state['download_buttons'].append(download_button_data)

            # Display the download button
            st.download_button(
                label=download_button_data['label'],
                data=download_button_data['text'],
                file_name=download_button_data['file_name'],
                key=download_button_data['id']
            )
    if save:
        # Save the message, including the download_id if available
        save_message(
            message,
            role,
            download_id=download_button_data['id'] if download_button_data else None
        )



def paint_message(message, role, download_id):
    with st.chat_message(role):
        st.markdown(message)
        download_data = None;
        if download_id:
            # Find the corresponding download button data
            download_data = next(
                (item for item in st.session_state['download_buttons'] if item['id'] == download_id),
                None
            )
        if download_data:
            st.download_button(
            label=download_data['label'],
                data=download_data['text'],
                file_name=download_data['file_name'],
                key=download_data['id']
            )

def paint_history():
    for message in st.session_state["messages"]:
        paint_message(
            message["message"],
            message["role"],
            message["download_id"]
        )




st.title("Assistants GPT")


if len(st.session_state["messages"])>0:
    paint_history()
else:
    st.markdown(
    """
    Welcome!
                
    Use this chatbot to ask any questions!
    """
    )
message = st.chat_input("Ask anything...")
if message:
    if st.session_state.get("APIKEY_failed", True) == False:
        client.api_key = api_key
        assistant = client.beta.assistants.create(
            name="Research Assistant",
            instructions="""You help users do research from the web. use the following tools to help the user and must save the content as a text file.
            1. Search the query using DuckDuckSearchTool.
            2. Search the query using WikiSearchTool.
            3. Save the content as text file.""",
            model=chat_model,
            tools=functions,
        )    
        print("message :",message)
        send_message(message, "human")
        thread = create_thread(message)
        print(thread.id, assistant.id)
        run = create_run(thread.id, assistant.id)
        print(get_run(run.id, thread.id).status,get_required_action(run.id, thread.id))    
        start_time = time.time()  # Record the start time
        previous_result = "None"
        while True:
            # todo : exception handling
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            if elapsed_time > 120:
                print("Timeout reached, stopping the polling.")
                break   
            run_status = get_run(run.id, thread.id).status
               
            if run_status == 'completed':
                print("Run completed successfully.")
                result = get_messages(thread.id)
                print(result)
                if(not(result["content"] == previous_result) and result["role"]== "assistant"):
                    previous_result = result["content"]
                    send_message(result["content"], "assistant")
                break
            elif run_status == 'failed':
                print("Run failed.")
            elif run_status == 'expired':
                print("Run was expired.")
            elif run_status == 'stopped':
                print("Run was stopped.")
            elif run_status == 'requires_action':
                print("Run is requires_action.")  
                run = get_run(run.id, thread.id)
                outputs = []
                for action in run.required_action.submit_tool_outputs.tool_calls:
                    action_id = action.id
                    function = action.function
                    # send_message(f"Calling function: {function.name} with arg {function.arguments}", "assistant",False)                        
                submit_tool_outputs(run.id, thread.id)
                count = count + 1
            elif run_status == 'queued':
                print("Run is queued.")
            elif run_status == 'in_progress':
                print("Run is in_progress.")
            elif run_status == 'cancelling':
                print("Run is cancelling.")
            elif run_status == 'cancelled':
                print("Run is cancelled.")
            print(f"Current run status: {run.status}, waiting for completion...")
            waiting_time = int((time.time() - start_time)*10)/10
            send_message(f"Current run status: {run.status}, waiting for completion... Waiting Time(sec):{waiting_time} / Time Out(sec):120", "ai",False)
            time.sleep(5)  # Wait for 5 seconds before checking again
        result = get_messages(thread.id)
        print(st.session_state["messages"])

        # send_message(result, "assistant")
        st.session_state['regenerate'] = False
    else:
        st.error("Invalid API Key. Please enter a valid API Key or Check API KEY")
        st.stop()
# else:
    # st.session_state["messages"] = []