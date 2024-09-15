from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import logging
from fake_useragent import UserAgent

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
# Configure logging
logging.basicConfig(level=logging.DEBUG)
# Initialize a UserAgent object
ua = UserAgent()
# @st.cache_data(show_spinner="Loading website...")
def load_website(url):
    try:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
        )
        loader = SitemapLoader(
            url,
            filter_urls=[
                r"^(.*\/blog\/).*",
            ],
            parsing_function=parse_page,
        )
        loader = SitemapLoader(url)
        loader.requests_per_second = 3
        # Set a realistic user agent
        loader.headers = {'User-Agent': ua.random}
        # docs = loader.load()
        docs = loader.load_and_split(text_splitter=splitter)
        logging.debug(f"Loaded documents: {docs}")
        return docs
    except Exception as e:
        logging.error(f"Error loading sitemap: {e}")
        return []


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)


st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)


with st.sidebar:
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
    )


if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        docs = load_website(url)
        st.write(docs)