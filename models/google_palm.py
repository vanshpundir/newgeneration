import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import UnstructuredCSVLoader
from langchain.chains import RetrievalQA

from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
import os, glob

# Sidebar contents
with st.sidebar:
    st.title("News App")
    st.markdown(
        """
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [PaLM](https://makersuite.google.com/app/home) Embeddings & LLM model

    """
    )
    add_vertical_space(5)
    st.write("Get Personalised News")

load_dotenv()


def main():
    st.header("Your Personalised News App 💬")

    files_path = "/Users/vansh/Desktop/newgeneration/newgeneration/data/concatenated_data.csv"
    loaders = [UnstructuredCSVLoader(files_path)]

    # if "index" not in st.session:
    index = VectorstoreIndexCreator(
        embedding=GooglePalmEmbeddings( google_api_key='AIzaSyD6Vlv1bz3QokAbLQTIswbz_afVvEQwxUo'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0),
    ).from_loaders(loaders)

    llm = GooglePalm(temperature=0.1,google_api_key= 'AIzaSyD6Vlv1bz3QokAbLQTIswbz_afVvEQwxUo')  # OpenAI()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.vectorstore.as_retriever(),
        # input_key="question",
        return_source_documents=True,
    )
    from langchain.csv_splitter import RecursiveCharacterCSVSplitter
    # st.session.index = index

    # Accept user questions/query
    query = st.text_input("Ask questions about your PDF file:")
    # st.write(query)
    if query:
        response = chain(query)
        st.write(response["result"])
        with st.expander("Returned Chunks"):
            for doc in response["source_documents"]:
                st.write(f"{doc.metadata['source']} \n {doc.page_content}")


if __name__ == "__main__":
    main()