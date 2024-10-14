from dotenv import load_dotenv
from langchain_teddynote import logging
import streamlit as st
from safetyhealth_chain import multi_modal_rag_chain, plt_image
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever, SearchType
from langchain_openai import OpenAIEmbeddings
import os
import pickle
from langchain_core.messages import ChatMessage

from PIL import Image, ImageDraw

# API KEY 정보로드
api_key = st.secrets["OPENAI_API_KEY"]

# 프로젝트 이름을 입력합니다.
logging.langsmith("YATech-v1")

# 사이트의 제목 입력
st.title("연암테크-ChatBot")

with st.sidebar:
    selected_category = st.selectbox("분야 선택", ["안전보건", "공정작업"], index=0)



# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")


# 대화 기록이 없다면, chat_history 라는 키로 빈 대화를 저장하는 list 를 생성
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# chain 을 초기화
if "rag_chain" not in st.session_state:
    st.session_state["rag_chain"] = None

# 대화 기록에 채팅을 추가
def add_history(role, message):
    st.session_state["chat_history"].append(ChatMessage(role=role, content=message))

def print_history():
    for chat_message in st.session_state["chat_history"]:
        # 메시지 출력(role: 누가 말한 메시지 인가?) .write(content: 메시지 내용)
        st.chat_message(chat_message.role).write(chat_message.content)





# 이전까지의 대화를 출력
print_history()


vectorstore = Chroma(
    persist_directory="./chroma_openai_1007_8.db",
    embedding_function=OpenAIEmbeddings(),
    collection_name="openai_1007_8",
)

with open("./chroma_openai_1007_8.pkl", "rb") as f:
    docstore = pickle.load(f)

id_key = "doc_id"
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    id_key=id_key,
)

retriever.search_type = SearchType.mmr
retriever.search_type = SearchType.similarity_score_threshold
retriever.search_kwargs = {"score_threshold": 0.5}
retriever.search_type = SearchType.similarity
retriever.search_kwargs = {"k": 5}



st.session_state["rag_chain"] = multi_modal_rag_chain(retriever)



user_input = st.chat_input("궁금한 내용을 입력해 주세요")


if user_input:
    rag_chain = st.session_state["rag_chain"]

    # 사용자의 질문을 출력
    st.chat_message("user").write(user_input)

    # AI의 질문을 출력
    with st.chat_message("ai"):
        # 답변을 출력할 빈 공간을 만든다.
        chat_container = st.empty()

        # 사용자가 질문을 입력하면, 체인에 질문을 넣고 실행합니다.
        answer = rag_chain.stream(user_input)

        # 스트리밍 출력
        ai_answer = ""
        for token in answer:
            ai_answer += token
            chat_container.markdown(ai_answer)


        # 임시 이미지 생성 및 출력
        image = plt_image(retriever, user_input)
        if image is not None:
            st.image(image, caption="Generated Test Image", use_column_width=True)



    # 대화 기록에 추가
    add_history("user", user_input)
    add_history("ai", ai_answer)


