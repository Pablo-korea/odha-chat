import os
import re
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import chainlit as cl
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage

rasi_analysis_prompt = """
라시차트 분석 시 다음 항목들을 포함해 정리해줘:

1. **각 행성의 하우스 위치, 별자리(Rasi), 세기(Strength)**  
2. **각 행성의 낙샤트라(Nakshatra)와 그 지배 행성 상태**  
3. **각 하우스별 의미 및 해당 하우스에 위치한 행성의 성격 분석**  
4. **하우스 세기(강/약), 빈 하우스에 대한 영향 해석**  
5. **케마드루마 요가, 니차/우차 상태, 점프라 요가 등 주요 요가 존재 여부**  
6. **라그나(Ascendant)의 위치와 전체 차트에서의 역할**  
7. **종합 해석: 주요 성격 경향, 삶의 기본 성향, 강점과 약점 요약**
"""

# 환경 변수 로딩
load_dotenv()

# 설정
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")
API_ENDPOINT = os.getenv("API_ENDPOINT")
API_ACCESS_TOKEN = os.getenv("API_ACCESS_TOKEN")
SYSTEM_PROMPT_PATH = os.getenv("SYSTEM_PROMPT_PATH", "prompt/system_prompt.txt")

llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST)

KNOWN_PLACES = [
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
    "김해", "수원", "창원", "전주", "포항"
]

def load_system_prompt(now_kst: str) -> str:
    try:
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().format(now_kst=now_kst)
    except:
        return f"지금은 {now_kst}입니다."

def resolve_gochra_datetime(text: str) -> str:
    now = datetime.now()
    text = text.lower()
    if "오늘" in text:
        return now.strftime("%Y-%m-%dT%H:%M:%S+09:00")
    elif "내일" in text:
        return (now + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S+09:00")
    elif "모레" in text:
        return (now + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S+09:00")
    elif "금요일" in text:
        weekday = now.weekday()
        days_ahead = (4 - weekday) % 7
        return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%dT%H:%M:%S+09:00")
    return now.strftime("%Y-%m-%dT%H:%M:%S+09:00")

def is_general_mode_request(text: str) -> bool:
    return "일반으로" in text or "api 없이" in text or "대화만" in text

def get_coordinates(place: str) -> str:
    try:
        res = requests.get("https://nominatim.openstreetmap.org/search", params={"q": place, "format": "json", "limit": 1}, headers={"User-Agent": "nowastro-chatbot"})
        data = res.json()
        if data:
            return f"{data[0]['lat']},{data[0]['lon']}"
    except:
        pass
    return "35.2721355,128.8452281"

def extract_gender(text: str) -> str | None:
    text = text.lower()
    for word in ["남성", "남자", "남", "man", "male", "boy"]:
        if word in text:
            return "남성"
    for word in ["여성", "여자", "여", "woman", "female", "girl"]:
        if word in text:
            return "여성"
    return None

def extract_place(text: str) -> str | None:
    match = re.search(r"([가-힣a-zA-Z]+)[\s]*(에서|출생|태어났|태어난)", text)
    if match:
        return match.group(1)
    for place in KNOWN_PLACES:
        if place in text:
            return place
    return None

def parse_and_store_user_info(user_input: str, session: dict) -> dict:
    result = session.copy()
    date_match = re.search(r"(\d{4})[년\-/. ]+(\d{1,2})[월\-/. ]+(\d{1,2})[일\s]*", user_input)
    time_match = re.search(r"(오전|오후)?\s*(\d{1,2})시(?:\s*(\d{1,2})분)?", user_input)

    if date_match:
        y, m, d = date_match.groups()
        hour, minute = "12", "00"
        if time_match:
            ampm, h, mm = time_match.groups()
            hour = str(int(h) + 12) if ampm == "오후" and int(h) < 12 else h
            minute = mm if mm else "00"
        result["datetime"] = f"{y}-{int(m):02d}-{int(d):02d}T{int(hour):02d}:{int(minute):02d}:00+09:00"

    place = extract_place(user_input)
    if place:
        result["place"] = place
        result["coordinates"] = get_coordinates(place)

    gender = extract_gender(user_input)
    if gender:
        result["usergender"] = gender

    result["birthdaytype"] = "음력" if "음력" in user_input else result.get("birthdaytype", "양력")
    result["gochradatetime"] = resolve_gochra_datetime(user_input)

    return result

def check_missing_fields(payload: dict):
    required = ["datetime", "place", "coordinates", "usergender"]
    missing = [k for k in required if k not in payload]
    questions = {
        "datetime": "⏰ 태어난 날짜와 시간을 알려주세요.",
        "place": "📍 태어난 지역을 알려주세요.",
        "coordinates": "🌐 출생지 정보를 더 정확히 알려주세요.",
        "usergender": "👤 성별을 알려주세요. (남성/여성)"
    }
    return [questions[m] for m in missing]

def call_astrology_api(payload: dict) -> str:
    headers = {
        "Authorization": f"Bearer {API_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    try:
        res = requests.post(API_ENDPOINT, json=payload, headers=headers, timeout=10)
        res.raise_for_status()
        return res.text
    except requests.exceptions.RequestException as e:
        return f"[API 요청 실패]: {str(e)}"

def is_astrology_query(text: str) -> bool:
    keywords = ["운세", "오늘", "점성", "궁합", "차트", "별자리", "행성", "출생", "사주", "궁도", "분석해 줘", "분석해주세요", "알려줘", "천문도", "라시", "점성학", "추천"]
    return any(k in text.lower() for k in keywords)

def is_rasi_analysis_request(text: str) -> bool:
    return any(k in text.lower() for k in ["라시차트", "천문도", "라그나", "차트를 분석"])

def is_short_response(text: str) -> bool:
    return len(text.strip()) < 10

def truncate_text(text: str, max_chars: int = 1500) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "..."


async def render_fortune_markdown(text: str, chunk_size: int = 120):
    import re

    text = text.strip()
    if not text:
        return

    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    for chunk in chunks:
        await cl.Message(content=chunk.strip(), author="운세봇").send()

async def proceed_with_astrology(payload: dict, user_input: str):
    now_kst = datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")
    system_prompt = SystemMessage(content=load_system_prompt(now_kst))
    api_result = call_astrology_api(payload)

    if is_rasi_analysis_request(user_input):
        short_result = truncate_text(api_result, 1500)
        prompt = f"사용자 질문: {user_input}\n\n출생 차트:\n{short_result}\n\n{rasi_analysis_prompt}"
    else:
        short_result = truncate_text(api_result, 1000)
        prompt = (
            f"사용자 질문: {user_input}\n\n"
            f"출생 차트 분석 결과:\n{short_result}\n\n"
            f"🔮 아래 형식에 따라 오늘의 운세를 통변해 주세요...\n"
        )
    try:
        print("1. 프롬프트 길이:", len(prompt))
        print("2. system_prompt:", system_prompt)
        print("3. prompt:", prompt)
        response = llm.invoke([system_prompt, HumanMessage(content=prompt)])
        print("4. response:", response.content)
        #await render_fortune_markdown(response.content)
    except Exception as e:
        await cl.Message(content=f"⚠️ LLM 응답 중 오류가 발생했습니다: {e}").send()

@cl.on_chat_start
async def start_chat():
    cl.user_session.set("user_info", {})
    cl.user_session.set("pending_fields", [])
    await cl.Message(content="🌟 안녕하세요! 생년월일, 출생시간, 출생지를 입력하시면 운세를 알려드릴게요. 다음처럼 질문해 주세요. 예)1976년 4월 27일 14시 서울에서 출생한 여자 현숙이의 라시차트를 분석해 줘").send()

@cl.on_message
async def handle(msg: cl.Message):
    user_input = msg.content.strip()
    session = cl.user_session.get("user_info") or {}
    pending = cl.user_session.get("pending_fields") or []

    if is_general_mode_request(user_input):
        cl.user_session.set("general_mode", True)
        await cl.Message(content="💬 이제부터는 일반 대화 모드입니다. API는 호출하지 않아요!").send()
        return

    if cl.user_session.get("general_mode", False):
        response = llm.invoke([HumanMessage(content=user_input)])
        await cl.Message(content=response.content).send()
        return

    if "기억된 정보" in user_input or "기억한 출생 정보" in user_input:
        if not session:
            await cl.Message(content="⚠️ 아직 입력된 출생 정보가 없습니다.").send()
        else:
            info = "\n".join([f"{k}: {v}" for k, v in session.items()])
            await cl.Message(content=f"🧠 현재 기억된 출생 정보는 다음과 같아요:\n{info}").send()
        return

    if is_short_response(user_input) and pending:
        updated = parse_and_store_user_info(user_input, session)
        cl.user_session.set("user_info", updated)
        cl.user_session.set("pending_fields", [])
        still_missing = check_missing_fields(updated)
        if still_missing:
            cl.user_session.set("pending_fields", still_missing)
            await cl.Message(content="\n".join(still_missing)).send()
        else:
            await proceed_with_astrology(updated, user_input)
        return

    if not is_astrology_query(user_input) and not is_rasi_analysis_request(user_input):
        now_kst = datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")
        system_message = SystemMessage(
            content=f"지금은 {now_kst}입니다. 이 시간 정보를 참고해서 질문에 정확하게 대답해 주세요."
        )
        response = llm.invoke([system_message, HumanMessage(content=user_input)])
        await cl.Message(content=response.content).send()
        return

    updated = parse_and_store_user_info(user_input, session)
    cl.user_session.set("user_info", updated)
    missing = check_missing_fields(updated)
    if missing:
        cl.user_session.set("pending_fields", missing)
        await cl.Message(content="\n".join(missing)).send()
        return

    cl.user_session.set("pending_fields", [])
    await proceed_with_astrology(updated, user_input)
