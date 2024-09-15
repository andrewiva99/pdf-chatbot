from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import (GoogleGenerativeAI, HarmBlockThreshold, HarmCategory)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_cohere import CohereEmbeddings
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os


def get_chat(get_session_history):
    load_dotenv()

    db_path = os.getenv('DB_PATH')
    cs_prompt_path = os.getenv('CS_PROMPT_PATH')
    s_prompt_path = os.getenv('S_PROMPT_PATH')

    llm = GoogleGenerativeAI(model='gemini-1.5-flash', temparture=0,
                             safety_settings=
                             {HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                              HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                              HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                              HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE},
                             stream=True)

    embeddings_model = CohereEmbeddings(model='embed-english-light-v3.0')

    vector_store = Chroma(embedding_function=embeddings_model, persist_directory=db_path)

    retriever = vector_store.as_retriever(k=10)

    with open(cs_prompt_path, 'r') as f:
        context_system_prompt = f.read()

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    with open(s_prompt_path, 'r') as f:
        system_prompt = f.read()

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain.pick('answer')
