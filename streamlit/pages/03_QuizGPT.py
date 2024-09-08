import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        print("Parsing!!!")
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make {question_count} questions to test the user's knowledge about the text.

    The difficulty of questions is {difficulty}    

    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
         
    Your turn!
         
    Context: {context}
""",
        )
    ]
)


formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a powerful formatting algorithm.
     
    You format exam questions into JSON format.
    Answers with (o) are the correct ones.
     
    Example Input:

    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Question: When was Avatar released?
    Answers: 2007|2001|2009(o)|1998
         
    Question: Who was Julius Caesar?
    Answers: A Roman Emperor(o)|Painter|Actor|Model
    
     
    Example Output:
     
    ```json
    {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }}
                ]
            }},
                        {{
                "question": "What is the capital or Georgia?",
                "answers": [
                        {{
                            "answer": "Baku",
                            "correct": false
                        }},
                        {{
                            "answer": "Tbilisi",
                            "correct": true
                        }},
                        {{
                            "answer": "Manila",
                            "correct": false
                        }},
                        {{
                            "answer": "Beirut",
                            "correct": false
                        }}
                ]
            }},
                        {{
                "question": "When was Avatar released?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }}
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                        {{
                            "answer": "A Roman Emperor",
                            "correct": true
                        }},
                        {{
                            "answer": "Painter",
                            "correct": false
                        }},
                        {{
                            "answer": "Actor",
                            "correct": false
                        }},
                        {{
                            "answer": "Model",
                            "correct": false
                        }}
                ]
            }}
        ]
     }}
    ```
    Your turn!

    Questions: {context}

""",
        )
    ]
)
@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    st.session_state[f"{file.name}_filecache_updated"] = True
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, difficulty_level):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    
    return chain.invoke({"context": format_docs(_docs),
                         "difficulty": difficulty_level,
                         "question_count": question_count[difficulty_level]})


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    st.session_state[f"{term}_wikicache_updated"] = True
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs

# Sidebar for API Key and GitHub Repo link
with st.sidebar:
    docs = None
    topic = None
    api_key = st.text_input("Enter your OpenAI API Key")
    if not api_key or st.session_state.get("APIKEY_failed"):
        st.error("Please enter your OpenAI API Key")
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    difficulty_level = st.selectbox("Select Difficulty", ["VeryEasy","Easy","Normal","Hard","Hell"])
    question_count = {
        "VeryEasy": 1,
        "Easy": 3,
        "Normal": 5,
        "Hard": 7,
        "Hell": 10
    }    
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
    else:
        topic = st.text_input("Search Wikipedia...")
        section = st.text_input("Type the section...(Future feature)")
    quiz_generating_btn = st.button("Generate Quiz")
    print("quiz_generating_btn status:",quiz_generating_btn)
 
    if quiz_generating_btn:
        llm = None
        # for checking if the API Key is valid
        try:
            llm = ChatOpenAI(
                api_key=api_key,
                temperature=0.1,
                model="gpt-3.5-turbo-1106",
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
            )
            llm.invoke("Hello!")   
            st.session_state["APIKEY_failed"] = False
            # You might want to add a simple test call here if the API does not get invoked elsewhere immediately
        except Exception as e:
            st.markdown(f"Failed to initialize ChatOpenAI: {str(e)}")   
            st.session_state["APIKEY_failed"] = True
            st.rerun()        
        if choice == "File":
            if file:
                docs = split_file(file)
            # todo: add a cache for the file upload
        else:
            if topic:
                print("Wiki Searching")
                docs = wiki_search(topic)
                print("Wiki Searching done")
            # todo: add a cache for the wiki search
            # todo: search for a specific section in the wiki article
    st.write("GitHub Repo:https://github.com/daegyujeong/gpt_pratice_dk/blob/c1e6fadad10a4afff5089f667c8704e6f0ab7f54/pages/03_QuizGPT.py")
       
        
if choice == "File":
    if file:
        quiz_cached = st.session_state.get(f"{file.name}_filecache_updated")
        print(f"cached_update : {file.name}_filecache_updated :",quiz_cached)
        if quiz_cached:
         docs = split_file(file)
    else:
        quiz_cached = False
else:        
    if topic:
        quiz_cached = st.session_state.get(f"{topic}_wikicache_updated")
        print(f"cached_update : {topic}_wikicache_updated",quiz_cached)
        if quiz_cached:
            docs = wiki_search(topic)
    else:
        quiz_cached = False

print("quiz_cached",quiz_cached,"quiz_generating_btn",quiz_generating_btn)
if quiz_generating_btn or quiz_cached:    
    if docs:
        print("start quiz generating")
        count = 0
        llm = ChatOpenAI(
            api_key=api_key,
            temperature=0.1,
            model="gpt-3.5-turbo-1106",
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )   
        questions_chain = questions_prompt | llm
        formatting_chain = formatting_prompt | llm
        topic = topic if topic else file.name
        response = run_quiz_chain(docs, topic, difficulty_level)
        print("start quiz displaying")
        question_form = st.form("questions_form")
        with question_form:
            correctcount = 0
            for question in response["questions"]:
                st.write(question["question"])
                value = st.radio(
                    "Select an option.",
                    [answer["answer"] for answer in question["answers"]],
                    index= None,
                    key = count,
                )
                count = count+1
                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("Correct!")
                    correctcount = correctcount + 1
                elif value is not None:
                    st.error("Wrong!")
                total_count = len(response["questions"])

            if st.session_state.get(f"{topic}_{difficulty_level}_corrected"):
                print("corrected")
                submit_button = st.form_submit_button("Retry")
                reset_button = st.form_submit_button("Reset Quiz")

            elif(quiz_cached):
                submit_button = st.form_submit_button("Retry")
            else:
                submit_button = st.form_submit_button("Submit Quiz")
            if submit_button:
                if correctcount == total_count:
                    st.write(f"Your Score is {correctcount}/{total_count}")
                    st.balloons()
                    print(st.session_state.get(f"{topic}_{difficulty_level}_corrected",False))
                    if st.session_state.get(f"{topic}_{difficulty_level}_corrected") == False:                      
                        reset_button = st.form_submit_button("Reset Quiz")
                        st.session_state[f"{topic}_{difficulty_level}_corrected"] = True
                else:
                    st.session_state[f"{topic}_{difficulty_level}_corrected"] = False
                    st.write(f"Your Score is {correctcount}/{total_count} Retry!")
 
        if st.session_state.get(f"{topic}_{difficulty_level}_corrected"):           
            if reset_button:
                st.session_state[f"{topic}_{difficulty_level}_corrected"] = False
                st.cache_data.clear()
                if choice == "File":
                    st.session_state[f"{topic}_filecache_updated"] = False
                    print(f"{topic}_filecache_updated false")
                else:               
                    st.session_state[f"{topic}_wikicache_updated"] = False
                    print(f"{topic}_wikicache_updated false")
                # del st.session_state['quiz_generating_btn']
                print("Cache Cleared")
                docs = None
                del question_form
                del submit_button
                del quiz_generating_btn
                st.rerun()
                # how can I clear specific cache?
                    
    if not docs:
            st.markdown(
                """
                Please Type Valid Input
            """
            )     
if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles to test your knowledge and help you study.
                
    Get started by searching on Wikipedia in the sidebar.
    """
    )        