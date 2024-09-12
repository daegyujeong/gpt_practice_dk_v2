import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from urllib.parse import urlparse
import os
import pickle

llm = ChatOpenAI(
    temperature=0.1,
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)

def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )

def parse_site_name(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    file_path = f"./.cache/sites/{parse_site_name(url)}.pkl"
    
    # Check if cache file exists
    if os.path.exists(file_path):
        # Load cached data
        with open(file_path, 'rb') as file:
            retriever = pickle.load(file)
    else:
        # Fetch data from the website
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=200,
        )
        loader = SitemapLoader(
            url,
            parsing_function=parse_page,
        )
        loader.requests_per_second = 2
        docs = loader.load_and_split(text_splitter=splitter)
        vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
        retriever = vector_store.as_retriever()

        # Save to cache
        with open(file_path, 'wb') as file:
            pickle.dump(retriever, file)

    return retriever


st.set_page_config(
    page_title="Cloudflare Documentation GPT",
    page_icon="🖥️",
)

st.markdown(
    """
    # Cloudflare Documentation GPT
            
    Ask questions about Cloudflare's AI Gateway, Cloudflare Vectorize, and Workers AI documentation.
"""
)

with st.sidebar:
    st.text_input("Your OpenAI API Key", key="openai_key")
    url = st.text_input(
        "Write down the Sitemap URL", value="https://developers.cloudflare.com/sitemap-0.xml"
    )
    st.markdown("[GitHub Repo](https://github.com/)")

message = st.chat_input("Ask anything about Cloudflare's products...")

if message:
    retriever = load_website(url)
    chain = (
        {
            "docs": retriever,
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(get_answers)
        | RunnableLambda(choose_answer)
    )
    result = chain.invoke(message)
    st.markdown(result.content.replace("$", "\$"))
