import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def load_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        max_output_tokens=50,
        top_p=0.95,
        top_k=40,
        api_key=os.getenv("GOOGLE_API_KEY"),
    )
    return llm

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

template = """
        Use the pieces of information provided in the context to answer user's question.
        If you dont know the answer, just say that you dont know, dont try to make up an answer. 
        Dont provide anything out of the given context

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """


def main():
    st.title("Medical Chatbot! : ")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    user_input = st.chat_input("Pass your prompt here")

    if user_input:
        st.chat_message('user').markdown(user_input)
        st.session_state.messages.append({'role':'user', 'content': user_input})
        
        
        try: 
            vectorstore=get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            llm = load_llm()

            parallel_chain = RunnableParallel({
                'context': retriever | RunnableLambda(format_docs),
                'question': RunnablePassthrough()
            })

            # use a different name so we don't shadow the user_input variable
            prompt_template = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )

            parser = StrOutputParser()

            # format the dict from parallel_chain into a string using prompt_template.format
            main_chain = parallel_chain | RunnableLambda(lambda out: prompt_template.format(context=out['context'], question=out['question'])) | llm | parser

            # invoke with the user input string (not the PromptTemplate object)
            response = main_chain.invoke(user_input)

            # handle response type safely (string vs dict)
            if isinstance(response, dict):
                result = response.get("result") or response.get("output") or str(response)
                source_documents = response.get("source_documents") or response.get("source_docs") or []
            else:
                result = str(response)
                source_documents = []

            result_to_show = result + "\nSource Docs:\n" + str(source_documents)
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error is : {str(e)}")

if __name__ == "__main__":
    main()
