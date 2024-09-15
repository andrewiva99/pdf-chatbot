import streamlit as st
from PyPDF2 import PdfReader
import json
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv, set_key
from load_save_memory import get_chat_paths, history_to_json, history_from_json, save_chat_paths
from langchain_community.chat_message_histories import ChatMessageHistory
from qa_chain import get_chat
import os


def get_files():
    with open(os.getenv('FILES_PATH'), 'r') as f:
        data = json.load(f)
    return data


def get_text_chunks(texts):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunk_text = text_splitter.split_text(texts)
    return chunk_text


def get_pdf_text(docs):
    pdf_store = {}
    for d in docs:
        reader = PdfReader(d)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        pdf_store[d.name] = text
    return pdf_store


def save_pdf(documents, files):
    file_names = files.keys()
    embeddings_model = CohereEmbeddings(model='embed-english-light-v3.0')
    vectorstore = Chroma(embedding_function=embeddings_model, persist_directory=os.getenv('DB_PATH'))
    for name, text in documents.items():

        if name in file_names:
            raise ValueError(f"A file named {name} already exists")

        chunk_text = get_text_chunks(text)

        ids = vectorstore.add_texts(chunk_text, metadatas=[{'pdf_name': name}] * len(chunk_text))
        files[name] = ids

    with open(os.getenv('FILES_PATH'), 'w') as f:
        json.dump(files, f)


def delete_pdf(selected, files):
    vectorstore = Chroma(persist_directory=os.getenv('DB_PATH'))
    for f in selected:
        ids = files.pop(f)
        vectorstore.delete(ids=ids)
    with open(os.getenv('FILES_PATH'), 'w') as f:
        json.dump(files, f)


@st.dialog("Confirm deletion of the following files:")
def submit_deletion(selected, files):
    st.write(selected)
    if st.button("Submit"):
        delete_pdf(selected, files)
        st.rerun()


def update_key():
    st.session_state.uploader_key += 1


def display_history(store, session_id):
    chat_history = store[session_id].messages
    for msg in chat_history:
        if msg.type == 'human':
            st.chat_message("user").write(msg.content)
        else:
            st.chat_message("assistant").write(msg.content)


def chat_bot(chat_id, chat_path):
    st.session_state.store = {chat_id: history_from_json(chat_path)}

    def session_history(session_id: str):
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    chain = get_chat(session_history)

    display_history(st.session_state.store, chat_id)
    user_input = st.chat_input("Ask a question")

    if user_input is not None:
        st.chat_message("user").write(user_input)

        config = {"configurable": {"session_id": chat_id}}

        st.chat_message('assistant').write_stream(chain.stream({'input': user_input}, config=config))
        history_to_json(st.session_state.store[chat_id], chat_path)


def create_chat(name):
    memory = ChatMessageHistory()
    file_name = name.replace(" ", "_")
    file_path = os.getenv("CHAT_HISTORIES_PATH") + '/' + file_name + "_history.json"
    history_to_json(memory, file_path)
    st.session_state.store_paths[name] = file_path
    save_chat_paths(st.session_state.store_paths, os.getenv("STORE_PATHS"))


def delete_chat(name):
    path = st.session_state.store_paths.pop(name)
    os.remove(path)
    save_chat_paths(st.session_state.store_paths, os.getenv("STORE_PATHS"))
    st.rerun()


def main():
    load_dotenv()

    files = get_files()
    file_names = files.keys()

    st.set_page_config(
        page_title="PDF Chatbot",
    )

    st.header('PDF Chatbot')

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    st.session_state.store_paths = get_chat_paths(os.getenv("STORE_PATHS"))

    with st.sidebar:

        name = st.text_input('Enter new chat name')
        if name is not None and st.button('Create chat'):
            create_chat(name)

        chat_names = st.session_state.store_paths.keys()

        chat_id = st.selectbox('Choose chat', chat_names, len(chat_names) - 1)

        if chat_id is not None and st.button('Delete chat', type='primary'):
            delete_chat(chat_id)

        with st.expander("Upload files"):

            docs = st.file_uploader(label='PDF uploader',
                                    type='pdf',
                                    accept_multiple_files=True,
                                    key=f'uploader_{st.session_state.uploader_key}')

            if st.button('Save') and docs is not None:
                try:
                    documents = get_pdf_text(docs)
                    save_pdf(documents, files)
                    update_key()
                    st.rerun()
                except ValueError as e:
                    st.error(f"ERROR: {e}")

        with st.expander("Delete files"):
            selected = st.multiselect("Choose .pdf file for delete", file_names, placeholder='Select files')
            if st.button('DELETE', type="primary") and len(selected) > 0:
                submit_deletion(selected, files)

        with st.expander("API Keys"):
            st.markdown("[Get your Google API KEY](https://console.cloud.google.com/apis/credentials)")
            new_key_google = st.text_input("Enter Google API KEY")
            if new_key_google is not None and st.button("Save Google API key"):
                set_key(".env", "GOOGLE_API_KEY", new_key_google)
                os.environ["GOOGLE_API_KEY"] = new_key_google

            st.markdown("[Get your Cohere API KEY](https://dashboard.cohere.com/api-keys)")
            new_key_cohere = st.text_input("Enter Cohere API KEY")
            if new_key_cohere is not None and st.button("Save Cohere API key"):
                set_key(".env", "COHERE_API_KEY", new_key_cohere)
                os.environ["COHERE_API_KEY"] = new_key_cohere

    google_key_unavailable = os.environ['GOOGLE_API_KEY'] == ""
    cohere_key_unavailable = os.environ['COHERE_API_KEY'] == ""

    if google_key_unavailable:
        st.warning("Enter your Google API Key in the 'API Keys' section")
    if cohere_key_unavailable:
        st.warning("Enter your Cohere API Key in the 'API Keys' section")

    if chat_id is not None and not google_key_unavailable and not cohere_key_unavailable:
        chat_bot(chat_id, st.session_state.store_paths[chat_id])


if __name__ == '__main__':
    main()
