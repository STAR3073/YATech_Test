from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from io import BytesIO
import io
import re
import base64
from IPython.display import HTML, display
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from PIL import Image


# base64 인코딩된 문자열을 이미지로 표시
def plt_img_base64(img_base64):
    image_html = f'<img src="{img_base64}" />'
    display(HTML(image_html))

# 문자열이 base64로 보이는지 확인
def looks_like_base64(sb):
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

# base64 데이터가 이미지인지 시작 부분을 보고 확인
def is_image_data(b64data):
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # 처음 8바이트를 디코드하여 가져옴
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    
    except Exception:
        return False

# Base64 문자열로 인코딩된 이미지의 크기 조정
def resize_base64_image(base64_string, size=(128, 128)):
    # Base64 문자열 디코드
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # 이미지 크기 조정
    resized_img = img.resize(size, Image.LANCZOS)

    # 조정된 이미지를 바이트 버퍼에 저장
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # 조정된 이미지를 Base64로 인코딩
    return base64.b64encode(buffered.getvalue()).decode("utf-8")



def plt_image(retriever, query):
    retrieved_docs = retriever.invoke(query)

    for doc in retrieved_docs:
        if isinstance(doc, Document):
            doc = doc.page_content

        if not looks_like_base64(doc) or not is_image_data(doc):
            continue

        try:
            base64_image = resize_base64_image(doc, size=(1300, 600))
            image_data = base64.b64decode(base64_image)
            image = Image.open(BytesIO(image_data))
            
            return image
        
        except Exception as e:
            continue

    return None





# base64로 인코딩된 이미지와 텍스트 분리
def split_image_text_types(docs):
    b64_images = []
    texts = []
    doc_titles = []
    for doc in docs:
        # 문서가 Document 타입인 경우 page_content 추출
        if isinstance(doc, Document):
            doc_titles.append(doc.metadata.get("doc_title"))
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts, "doc_titles": doc_titles}

# 컨텍스트를 단일 문자열로 결합
def img_prompt_func(data_dict):
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    doc_titles = "\n".join(data_dict["context"]["doc_titles"])
    messages = []

    # 이미지가 있으면 메시지에 추가
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    # 분석을 위한 텍스트 추가
    text_message = {
        "type": "text",
        "text": (
            "당신은 신입사원들에게 작업자의 안전 및 건강에 대한 교육을 진행하는 강사입니다.\n"
            "당신에게 제공되는 자료는 텍스트와 이미지가 혼합되어 있습니다.\n"
            "당신에게 제공되는 자료를 참고하여 사용자의 질문과 관련된 안전 및 건강에 대한 답변을 제공하세요.\n"
            "답변은 반드시 한국어로 해야합니다.\n"
            f"사용자 질문: {data_dict['question']}\n\n"
            f"참고할 자료: {formatted_texts}\n\n"

            "아래의 답변 예시를 참고하여 답변을 생성하세요.\n"
            "답변 예시:\n"
            "'안전 관리 기본 지침서'에 따르면, 작업 환경에서의 안전을 유지하기 위해서는 개인 보호 장비(PPE)를 항상 착용해야 합니다. "
            "또한, 사고를 예방하기 위해 정기적인 장비 점검과 유지 관리가 필요합니다."
            f"해당 답변은 {doc_titles} 문서를 참고하여 작성되었습니다.\n"            

        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]

# Multimodal RAG 체인
def multi_modal_rag_chain(retriever):
    # Multimodal LLM
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", max_tokens=2048)
    
    # RAG 파이프라인
    chain = (
        {"context": retriever | RunnableLambda(split_image_text_types),
         "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )

    return chain





### QUIZ Chain 생성하기


# 컨텍스트를 단일 문자열로 결합
def img_quiz_prompt_func(data_dict):
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # 이미지가 있으면 메시지에 추가
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    # 분석을 위한 텍스트 추가
    text_message = {
        "type": "text",
        "text": (
            "당신은 신입사원들에게 작업자의 안전 및 건강에 대한 교육을 진행하는 강사입니다.\n"
            "당신의 임무는 당신에게 제공되는 자료를 활용하여 신입사원들을 위한 퀴즈(quiz)를 만드는 것입니다.\n"
            "당신에게 제공되는 자료는 텍스트와 이미지가 혼합되어 있습니다.\n"
            "퀴즈는 4지선 다형 객관식 문제로 만들어 주세요. 문항은 3문항을 만들어 주세요.\n"
            "문항의 각 난이도는 쉬움, 보통, 어려움으로 나누어 주세요.\n"
            "출력은 반드시 한국어로 해야합니다.\n"
            f"User-provided question: {data_dict['question']}\n\n"
            "참고할 자료:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


# Multimodal RAG 체인
def multi_modal_rag_quiz_chain(retriever):
    # Multimodal LLM
    model = ChatOllama(temperature=0, model="llava-llama3:latest", max_tokens=2048)
    
    # RAG 파이프라인
    chain = (
        {"context": retriever | RunnableLambda(split_image_text_types),
         "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_quiz_prompt_func)
        | model
        | StrOutputParser()
    )

    return chain