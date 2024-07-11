import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
# from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
import google.generativeai as genai
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from PIL import Image

from dotenv import load_dotenv

load_dotenv()
# groq_api_key=os.getenv('GROQ_API_KEY')
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def get_pdf_text():
    text=""
    pdf_reader= PdfReader("Bharatiya_Nyaya_Sanhita_2023.pdf")
    for page in pdf_reader.pages:
        text+= page.extract_text()
    return text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    This is your introduction - Your name is " Sahi Jawab (Your Nyaya Mitra) " and you are developed by "Keshav Agrawal".

    You're a go-to platform for all the legal queries. You are embedded with the entire data of the three newly enacted criminal laws namely - The Bharatiya Nyaya Sanhita (BNS), the Bharatiya Nagrik Suraksha Sanhita (BNSS), and the Bharatiya Sakshya Adhiniyam (BSA) to provide accurate and reliable information on Indian laws.

    Your aim is to make legal knowledge accessible to everyone. Simply user will ask their questions, and you will guide them with clear and concise answers. 

    Whether they are seeking legal advice or just curious about the law, you are there to help. 

    Use suitable emojis wherever needed.

    Greet them with Radhe Radhe 🙏

    Generally, user starts with a greeting first. So, greet them accordingly, and ask them for their queries.

    You'll never use any muslim words in your conversation.

    If user asks anything about yourself, then answer them with polite words. don't give very straight forward one liner answers.

    Ensure to provide suitable answers - if the answer demands more detail, provide it, but don't give lengthy answers unnecessarily.
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{input}\n

    Answer:
    """

    # model = ChatGroq(groq_api_key=groq_api_key,
    #                 model_name="Llama3-8b-8192",
    #                 temperature=0.7)

    model = Ollama(model="llama3")

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question": user_question})
        # , return_only_outputs=True)

    # st.write(response)
    # st.write("Reply: ", response["output_text"])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["input_documents"]):
            st.write(doc.page_content)
            st.write("--------------------------------")

    return response["output_text"]





def main():
    
    st.set_page_config(
        page_title='Sahi Jawab', 
        layout='wide',
        page_icon="⚖️"               
    )

    # st.header("Sahi Jawab : Your Nyaya Mitra 👩🏻‍⚖️📚𓍝🏛️")

    st.sidebar.title("Sahi Jawab : Your Nyaya Mitra")

    # st.image("Sahi Jawab.png", use_column_width=True,caption='Sahi Jawab')

    # Page Setup 
    #Image In Sidebar 

    # st.logo(sidebar_logo, icon_image=main_body_logo)  format

    st.logo("sidebar_logo.png", icon_image="only_logo.png")
    
    with st.sidebar.container(): 
        st.image('Sahi Jawab.png', use_column_width=True, caption='Sahi Jawab : Your Nyaya Mitra 👩🏻‍⚖️📚𓍝')
        with st.expander("About Us",icon=":material/info:"):
            st.success("Hii, I am your go-to platform for all your legal queries. We have embedded the entire Bhartiya Nyaya Sanhita to provide accurate and reliable information on Indian laws. Our aim is to make legal knowledge accessible to everyone. Simply ask your questions, and our intelligent system will guide you with clear and concise answers. Whether you're seeking legal advice or just curious about the law, Sahi Jawab is here to help.")
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

    # st.subheader("Ask your question : ")
    # user_question = st.text_input("Ask your Question :",label_visibility="collapsed")
    # user_question = st.chat_input()
    # ask=st.button("Let the Magic Begin !! ")

    

    if user_question := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)
        # user_input(user_question)

    # Generate a new response if last message is not from assistant

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(user_question)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)

        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)



    with st.sidebar:
        st.title("Start the App by Clicking Here ✅")
        if st.button("Start Documents Embedding"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text()
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.info("VectorDB Store is Ready")
                st.success("You're good to go !! ")
                st.success("Ask Questions now...")
        st.sidebar.write("---\n")
        st.sidebar.success(print_praise())   
        st.sidebar.write("---\n")
        st.sidebar.info("Special Thanks to our Mentor\n\nDr.Ankur Rai, Professor, \n\nGLA UNIVERSITY, Mathura")
        st.sidebar.write("---\n")


    # # Store LLM generated responses
    # if "messages" not in st.session_state.keys():
    #     st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]



    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    st.sidebar.write("---\n")

    st.sidebar.markdown(
        "<h3 style='text-align: center;'>Developed with ❤️ for GenAI by <a style='text-decoration: none' href='https://www.linkedin.com/in/keshavagrawal595/'>Keshav Agrawal</a></h3>",
        unsafe_allow_html=True
    )

    # if st.session_state.messages[-1]["role"] != "assistant":
    #     with st.chat_message("assistant"):
    #         with st.spinner("Thinking..."):
    #             response = user_input(user_question)
    #             # placeholder = st.empty()
    #             # full_response = ''
    #             # for item in response:
    #             #     full_response += item
    #             #     placeholder.markdown(full_response)
    #             # placeholder.markdown(full_response)
    #     message = {"role": "assistant", "content": response}
    #     st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
