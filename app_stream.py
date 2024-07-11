import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import google.generativeai as genai
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

load_dotenv()

global model
global groq_api_key
global google_api_key

groq_api_key=os.getenv('GROQ_API_KEY') 

def load_model():
    global model

    genai.configure(api_key=google_api_key)

    model = ChatGroq(groq_api_key=groq_api_key,
                        model_name="Llama3-70b-8192")
    

prompt_template = ChatPromptTemplate.from_template(
"""
This is your introduction - Your name is " Sahi Jawab (Your Nyaya Mitra) " and you are developed by "Keshav Agrawal".

You're a go-to platform for all the legal queries. You are embedded with the entire data of the three newly enacted criminal laws namely - The Bharatiya Nyaya Sanhita (BNS), the Bharatiya Nagrik Suraksha Sanhita (BNSS), and the Bharatiya Sakshya Adhiniyam (BSA) to provide accurate and reliable information on Indian laws.

Your aim is to make legal knowledge accessible to everyone. Simply user will ask their questions, and you will guide them with clear and concise answers. 

Whether they are seeking legal advice or just curious about the law, you are there to help. 

Use suitable emojis wherever needed.

Greet them with Radhe Radhe 🙏

Generally, user starts with a greeting first. So, greet them accordingly, and ask them for their queries.

You'll never use any arabic words in your conversation.

If user asks anything about yourself, then answer them with polite words. don't give very straight forward one liner answers.

Ensure to provide suitable answers - if the answer demands more detail, provide it, but don't give lengthy answers unnecessarily.
Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
Context:\n {context}?\n
Question: \n{input}\n

Answer:
"""
)

def get_vector_store(progress):
    global google_api_key

    if "vectors" not in st.session_state:

        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        progress.progress(10)

        st.session_state.loader=PyPDFDirectoryLoader("./bns") ## Data Ingestion
        st.session_state.docs=st.session_state.loader.load() ## Document Loading
        progress.progress(30)

        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs) #splitting
        progress.progress(60)

        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector Google embeddings
        progress.progress(100)
    
def main():
    global model
    global groq_api_key
    global google_api_key


    st.set_page_config(
        page_title='Sahi Jawab', 
        layout='wide',
        page_icon="⚖️"               
    )
    st.sidebar.title("Sahi Jawab : Your Nyaya Mitra")
    st.logo("logo/sidebar_logo.png", icon_image="logo/only_logo.png")
    with st.sidebar.container(): 
        st.image('logo/Sahi Jawab.png', use_column_width=True, caption='Sahi Jawab : Your Nyaya Mitra 👩🏻‍⚖️📚𓍝')
        with st.expander("About Us",icon=":material/info:"):
            st.success("Hii, I am your go-to platform for all your legal queries. We have embedded the entire Bhartiya Nyaya Sanhita to provide accurate and reliable information on Indian laws. Our aim is to make legal knowledge accessible to everyone. Simply ask your questions, and our intelligent system will guide you with clear and concise answers. Whether you're seeking legal advice or just curious about the law, Sahi Jawab is here to help.")
        st.sidebar.markdown("---")



    # GROQ & Gemini Credentials
    with st.sidebar:
        if 'GROQ_API_KEY' in st.secrets:
            st.success('API key already provided!', icon='✅')
            groq_api_key = st.secrets['GROQ_API_KEY']
        else:
            groq_api_key = st.text_input('Enter GROQ API Key:', type='password')
            if not (groq_api_key.startswith('gsk_') and len(groq_api_key)==56):
                st.warning('Please enter your credentials!', icon='⚠️')
            else:
                os.environ['GROQ_API_KEY']=groq_api_key
                st.success('Proceed to entering your prompt message!', icon='👉')

        if 'GOOGLE_API_KEY' in st.secrets:
            st.success('API key already provided!', icon='✅')
            google_api_key = st.secrets['GOOGLE_API_KEY']
        else:
            google_api_key = st.text_input('Enter Google API Key:', type='password')
            if not (google_api_key):
                st.warning('Please enter your credentials!', icon='⚠️')
            else:
                os.environ['GOOGLE_API_KEY']=google_api_key
                st.success('Proceed to entering your prompt message!', icon='👉')
    if groq_api_key and google_api_key:
        load_model()
    else:
        st.write("Enter valid api keys")

    st.sidebar.markdown("---")

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    def print_praise():
        praise_quotes = """
        Keshav Agrawal
    2nd year Student
    B.Tech(Hons) CSE AI-ML
        """
        title = "**Developed By -**\n\n"
        return title + praise_quotes

    with st.sidebar:
        st.title("Start the App by Clicking Here ✅")
        doc=st.button("Start Documents Embedding")
        
        
        if doc or st.session_state.get('embedding_done', False):
            if google_api_key and groq_api_key:
                with st.spinner("Processing..."):
                    progress_bar = st.progress(0)
                    get_vector_store(progress_bar)
                    st.info("VectorDB Store is Ready")
                    st.success("You're good to go !! ")
                    st.success("Ask Questions now...")
                    st.session_state.embedding_done = True
            else:
                st.error("Please Enter API Keys first")

        st.write("---\n")
        st.success(print_praise())   
        st.write("---\n")
        st.info("Special Thanks to our Mentor\n\nDr.Ankur Rai, Professor, \n\nGLA UNIVERSITY, Mathura")
        st.write("---\n")

    user_question = st.chat_input(disabled=not (groq_api_key and google_api_key and st.session_state.get('embedding_done', False)))
    
    if user_question:

        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

    import time 

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                document_chain=create_stuff_documents_chain(model,prompt_template)
                retriever=st.session_state.vectors.as_retriever()
                retrieval_chain=create_retrieval_chain(retriever,document_chain)

                start=time.process_time()

                response=retrieval_chain.invoke({'input':user_question})

                print("Response time :",time.process_time()-start)

                st.write(response['answer'])

                # With a streamlit expander
                with st.expander("Document Similarity Search"):
                    # Find the relevant chunks
                    for i, doc in enumerate(response["context"]):
                        st.write(doc.page_content)
                        st.write("--------------------------------")    

        message = {"role": "assistant", "content": response['answer']}
        st.session_state.messages.append(message)
    st.sidebar.title("Looking to Restart your Conversation 🔄")
    st.sidebar.button('Start a New Chat', on_click=clear_chat_history)

    st.sidebar.write("---\n")

    st.sidebar.markdown(
        "<h3 style='text-align: center;'>Developed with ❤️ for GenAI by <a style='text-decoration: none' href='https://www.linkedin.com/in/keshavagrawal595/'>Keshav Agrawal</a></h3>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
