import streamlit as st
import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
import pandas as pd

def add_vertical_space(spaces=1):
    for _ in range(spaces):
        st.sidebar.markdown("---")


def main():
    
    st.set_page_config(page_title="Llama-2-GGML Desi Rail Chatbot")
    st.title("Llama-2-GGML Desi Rail Chatbot")

    st.sidebar.title("About")
    st.sidebar.markdown('''
        The Llama-2-GGML CSV Chatbot uses the **Llama-2-7B-Chat-GGML** model.
        
        ### 🔄Bot evolving, stay tuned!
        
        ## Useful Links 🔗
        
        - **Model:** [Llama-2-7B-Chat-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main) 📚
        - **GitHub:** [ThisIs-Developer/Llama-2-GGML-CSV-Chatbot](https://github.com/ThisIs-Developer/Llama-2-GGML-CSV-Chatbot) 💬
    ''')

   
    
    

    def processing():
        # specify the location of the d2.csv file below
        uploaded_file=**'C:\\Users\\New folder\\d2.csv'**
        
        # vectorestore folder path
        DB_FAISS_PATH = "vectorstore/db_faiss"
        add_vertical_space(1)
        
        st.sidebar.write('Made by [@ThisIs-Developer](https://huggingface.co/ThisIs-Developer)')

        if uploaded_file is not None:
        
            file_path=**'C:\\Users\\New folder\\d2.csv'**
        
            st.write("Processing available information...")
        


            loader = CSVLoader(file_path=file_path, encoding="utf-8", csv_args={'delimiter': ','})
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)#500,20
            text_chunks = text_splitter.split_documents(data)

            st.write(f"Total text chunks: {len(text_chunks)}")

            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            docsearch = FAISS.from_documents(text_chunks, embeddings)
            docsearch.save_local(DB_FAISS_PATH)

            llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
                                model_type="llama",
                                max_new_tokens=512,
                                temperature=0.2)
            


            qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

            st.write("Enter your query:")
            prompt=st.text_input("Input Prompt:")
            
            if prompt:
                with st.spinner("Processing your question..."):
                    chat_history = []
                    result = qa({"question": prompt, "chat_history": chat_history})
                    st.write("Response:", result['answer'])
                   
                    
            os.remove(file_path)
            
    def initialize(x,y):
        
        with st.form(key='my_form'):
            
            
            df1=df[(df['Source Station Name'] == x) & (df['Destination Station Name'] == y)]
            df1[['Train No', 'Train Name', 'Sequences','Station Name', 'Next Station','Arrival time', 'Departure Time', 'Distance1']].to_csv('d2.csv',encoding='utf-8')
            submit_button = st.form_submit_button(label='Chat')

            if submit_button:
                st.session_state['key'] =processing()

    
    df=pd.read_csv('cleaned_train.csv')
    start=st.selectbox('Choose Source station',list(df['Source Station Name'].unique()),key='my_slider',index=None,placeholder= "Choose an option")
    if start:
        stop=st.selectbox('Choose Destination station',list(df[df['Source Station Name']==start]['Destination Station Name'].unique()), key='my_checkbox',index=None,placeholder= "Choose an option")
    
    # Running the program
        initialize(start,stop)

if __name__ == "__main__":
    main()
