import re
from pathlib import Path
from collections import Counter
from urllib.parse import urlparse, parse_qs
from itertools import combinations

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="보컬로이드 감성 분석 및 노래 추천 웹앱",
    page_icon="🎧",
    layout="wide"
)

MOOD_COLS = [
    "joy", "sadness", "dreamy", "tension",
    "dark", "energy", "cute", "chaotic"
]

MOOD_KO = {
    "joy": "기쁨",
    "sadness": "슬픔",
    "dreamy": "몽환",
    "tension": "긴장감",
    "dark": "어두움",
    "energy": "에너지",
    "cute": "귀여움",
    "chaotic": "혼란스러움"
}

MOOD_EXPLAIN = {
    "joy": "밝고 긍정적인 정서",
    "sadness": "슬픔, 우울, 상실감",
    "dreamy": "몽환적이고 아련한 분위기",
    "tension": "긴장감, 압박감, 불안감",
    "dark": "어둡고 무거운 정서",
    "energy": "속도감과 활력",
    "cute": "귀엽고 가벼운 인상",
    "chaotic": "혼란, 광기, 예측 불가능성"
}

TAG_MOOD_RULES = {
    "happy":      {"joy": 30,  "sadness": -10, "dreamy": 0,  "tension": 0,   "dark": -10, "energy": 20,  "cute": 10, "chaotic": 0},
    "sad":        {"joy": 0,   "sadness": 30,  "dreamy": 5,  "tension": 0,   "dark": 10,  "energy": -10, "cute": 0,  "chaotic": 0},
    "dreamy":     {"joy": 0,   "sadness": 5,   "dreamy": 35, "tension": 0,   "dark": 5,   "energy": -5,  "cute": 0,  "chaotic": 0},
    "dark":       {"joy": -10, "sadness": 15,  "dreamy": 5,  "tension": 10,  "dark": 35,  "energy": 0,   "cute": 0,  "chaotic": 10},
    "cute":       {"joy": 20,  "sadness": 0,   "dreamy": 5,  "tension": 0,   "dark": 0,   "energy": 10,  "cute": 35, "chaotic": 0},
    "energetic":  {"joy": 10,  "sadness": -5,  "dreamy": 0,  "tension": 10,  "dark": 0,   "energy": 35,  "cute": 0,  "chaotic": 10},
    "calm":       {"joy": 5,   "sadness": 5,   "dreamy": 20, "tension": -15, "dark": 0,   "energy": -20, "cute": 0,  "chaotic": -10},
    "angry":      {"joy": -5,  "sadness": 10,  "dreamy": 0,  "tension": 30,  "dark": 20,  "energy": 35,  "cute": 0,  "chaotic": 25},
    "lonely":     {"joy": 0,   "sadness": 25,  "dreamy": 15, "tension": 0,   "dark": 15,  "energy": -10, "cute": 0,  "chaotic": 0},
    "chaotic":    {"joy": 0,   "sadness": 0,   "dreamy": 0,  "tension": 20,  "dark": 10,  "energy": 25,  "cute": 0,  "chaotic": 35},
    "nostalgic":  {"joy": 5,   "sadness": 20,  "dreamy": 20, "tension": 0,   "dark": 5,   "energy": -10, "cute": 0,  "chaotic": 0},
    "funny":      {"joy": 30,  "sadness": -10, "dreamy": 0,  "tension": 0,   "dark": -10, "energy": 20,  "cute": 15, "chaotic": 20},
    "romantic":   {"joy": 20,  "sadness": 10,  "dreamy": 15, "tension": 0,   "dark": 0,   "energy": 0,   "cute": 10, "chaotic": 0},
    "horror":     {"joy": -20, "sadness": 10,  "dreamy": 10, "tension": 30,  "dark": 40,  "energy": 5,   "cute": 0,  "chaotic": 20},
    "mysterious": {"joy": -5,  "sadness": 5,   "dreamy": 25, "tension": 15,  "dark": 20,  "energy": 0,   "cute": 0,  "chaotic": 10},
    "hopeful":    {"joy": 25,  "sadness": -5,  "dreamy": 10, "tension": 0,   "dark": -10, "energy": 15,  "cute": 5,  "chaotic": 0},
    "despair":    {"joy": -20, "sadness": 35,  "dreamy": 5,  "tension": 15,  "dark": 35,  "energy": -5,  "cute": 0,  "chaotic": 10},
    "fast":       {"joy": 5,   "sadness": -5,  "dreamy": 0,  "tension": 10,  "dark": 0,   "energy": 30,  "cute": 0,  "chaotic": 10},
    "slow":       {"joy": 0,   "sadness": 10,  "dreamy": 15, "tension": -10, "dark": 5,   "energy": -25, "cute": 0,  "chaotic": -10},
    "electronic": {"joy": 10,  "sadness": 0,   "dreamy": 15, "tension": 10,  "dark": 0,   "energy": 20,  "cute": 0,  "chaotic": 10},
    "rock":       {"joy": 0,   "sadness": 5,   "dreamy": 0,  "tension": 15,  "dark": 10,  "energy": 25,  "cute": 0,  "chaotic": 10},
    "ballad":     {"joy": 5,   "sadness": 20,  "dreamy": 10, "tension": -10, "dark": 0,   "energy": -20, "cute": 0,  "chaotic": -10},
}

TAG_DESCRIPTIONS = {
    "happy": "행복 기쁨 즐거움 밝음 유쾌함 긍정적인 감정 산뜻한 분위기 웃음 신나는 하루",
    "sad": "슬픔 우울 눈물 상실감 아픔 후회 무너짐 마음이 가라앉는 느낌 쓸쓸한 정서",
    "dreamy": "몽환 꿈 환상 새벽 멍함 흐릿함 공중에 뜬 느낌 아련함 비현실적 감각",
    "dark": "어두움 암울함 불안 죽음 무거움 침잠 공포 직전의 분위기 그림자 같은 정서",
    "cute": "귀여움 깜찍함 가벼움 사랑스러움 밝고 통통 튀는 느낌 캐릭터성이 강한 분위기",
    "energetic": "에너지 활기 강렬함 폭발 빠른 템포 질주감 고조되는 감정 신나는 리듬",
    "calm": "차분함 잔잔함 편안함 조용함 안정적인 새벽에 듣기 좋은 멍하니 듣는 분위기",
    "angry": "분노 짜증 화남 반항 공격적인 감정 날카로움 억눌린 감정의 폭발",
    "lonely": "외로움 고독 혼자 공허함 쓸쓸함 아무도 없는 밤 거리 고립된 감정",
    "chaotic": "혼란 정신없음 광기 난장판 붕괴 산만함 머리가 터질 것 같은 복잡함",
    "nostalgic": "추억 그리움 옛날 기억 회상 지나간 시간 어린 시절 아련한 과거",
    "funny": "웃김 개그 장난 바보 같음 엉뚱함 가벼운 농담 밈 같은 분위기",
    "romantic": "사랑 연애 설렘 고백 관계 감정적인 애정 부드러운 로맨스",
    "horror": "공포 무서움 괴담 기괴함 불쾌감 서늘함 무언가 잘못된 느낌",
    "mysterious": "미스터리 수상함 비밀 알 수 없음 신비로움 정체를 모르겠는 분위기",
    "hopeful": "희망 앞으로 나아감 괜찮아질 것 같은 느낌 회복 긍정적인 미래",
    "despair": "절망 포기 끝 무력감 무너짐 살기 힘듦 벼랑 끝에 몰린 감정",
    "fast": "빠름 속도감 질주 달려감 빠른 비트 몰아치는 전개",
    "slow": "느림 천천히 느긋함 늘어짐 여백 있는 호흡 느린 템포",
    "electronic": "전자음 기계적 디지털 신스 사이버 보컬로이드다운 차가운 질감",
    "rock": "록 락 기타 밴드 강한 드럼 거친 사운드 폭발적인 악기",
    "ballad": "발라드 감성적 느린 노래 서정적 멜로디 감정선을 따라가는 노래",
}

TEXT_KEYWORDS = {
    "happy": ["행복", "기쁨", "즐거", "신나", "밝은", "유쾌", "희망", "happy", "fun"],
    "sad": ["슬픔", "슬픈", "우울", "눈물", "울적", "상처", "sad", "blue"],
    "dreamy": ["몽환", "꿈", "환상", "아련", "신비", "새벽", "멍", "dream"],
    "dark": ["어두", "암울", "불안", "죽음", "무거운", "dark"],
    "cute": ["귀여", "귀여운", "깜찍", "cute", "kawaii"],
    "energetic": ["에너지", "활기", "강렬", "폭발", "신나는", "energetic"],
    "calm": ["잔잔", "차분", "편안", "조용", "calm"],
    "angry": ["분노", "화남", "화나는", "짜증", "angry"],
    "lonely": ["외로", "고독", "쓸쓸", "혼자", "lonely"],
    "chaotic": ["혼란", "정신없", "광기", "난장판", "chaotic"],
    "nostalgic": ["추억", "그리움", "옛날", "nostalgic"],
    "funny": ["웃긴", "개그", "장난", "바보", "funny"],
    "romantic": ["사랑", "연애", "로맨스", "romantic"],
    "horror": ["공포", "무서", "괴담", "horror"],
    "mysterious": ["미스터리", "수상", "비밀", "mysterious"],
    "hopeful": ["희망", "앞으로", "괜찮", "hope"],
    "despair": ["절망", "포기", "끝", "despair"],
    "fast": ["빠른", "질주", "속도", "fast"],
    "slow": ["느린", "천천히", "slow"],
    "electronic": ["전자", "기계", "디지털", "electronic"],
    "rock": ["락", "록", "기타", "rock"],
    "ballad": ["발라드", "ballad"],
}


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.35rem;
        font-weight: 850;
        margin-bottom: 0.2rem;
        letter-spacing: -0.04em;
    }
    .sub-text {
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }
    .song-card {
        border: 1px solid #e7e7e7;
        border-radius: 15px;
        padding: 17px 19px;
        margin-bottom: 12px;
        background-color: #ffffff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.045);
    }
    .song-title {
        font-size: 1.18rem;
        font-weight: 850;
        margin-bottom: 6px;
        letter-spacing: -0.03em;
    }
    .small-muted {
        color: #777;
        font-size: 0.9rem;
    }
    .tag-pill {
        display: inline-block;
        padding: 4px 9px;
        border-radius: 999px;
        background-color: #f1f1f1;
        margin: 2px;
        font-size: 0.82rem;
    }
    .analysis-box {
        border: 1px solid #ededed;
        border-radius: 14px;
        padding: 15px 17px;
        background-color: #fbfbfb;
        margin-bottom: 12px;
    }
    .rank-button-note {
        font-size: 0.85rem;
        color: #777;
        margin-bottom: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 데이터 로드
# =========================================================

def find_data_path():
    candidates = [
        Path("/data/songs_input.csv"),
        Path("data/songs_input.csv"),
        Path("songs_input.csv"),
    ]

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


@st.cache_data
def load_songs():
    path = find_data_path()

    if not path.exists():
        st.error(
            "songs_input.csv 파일을 찾지 못했습니다. "
            "현재 기준으로 /data/songs_input.csv 또는 data/songs_input.csv 위치에 파일을 두세요."
        )
        st.stop()

    df = pd.read_csv(path, dtype={"song_id": str})

    if "producer" in df.columns and "producers" not in df.columns:
        df["producers"] = df["producer"]

    if "vocal_engine" in df.columns and "vocal_engines" not in df.columns:
        df["vocal_engines"] = df["vocal_engine"]

    required_cols = [
        "song_id", "title_original", "title_ko",
        "producers", "vocal_engines",
        "main_link", "mood_tags"
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    for col in MOOD_COLS:
        if col not in df.columns:
            df[col] = 50
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(50)

    df = df.dropna(subset=["song_id"]).copy()
    df["song_id"] = df["song_id"].astype(str)

    return df


songs = load_songs()


# =========================================================
# 공통 함수
# =========================================================

def split_values(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value or value.lower() == "nan":
        return []

    return [x.strip() for x in value.split(",") if x.strip()]


def join_display(value):
    items = split_values(value)
    return ", ".join(items) if items else "-"


def count_split_column(df, column):
    counter = Counter()

    if column not in df.columns:
        return pd.DataFrame(columns=["name", "count"])

    for value in df[column].dropna():
        for item in split_values(value):
            counter[item] += 1

    result = pd.DataFrame(counter.items(), columns=["name", "count"])

    if result.empty:
        return result

    return result.sort_values("count", ascending=False).reset_index(drop=True)


def get_vector(df):
    return df[MOOD_COLS].astype(float).to_numpy()


def minmax_rescale_matrix(matrix):
    matrix = np.asarray(matrix, dtype=float)

    mins = matrix.min(axis=0)
    maxs = matrix.max(axis=0)
    denom = np.where(maxs - mins == 0, 1, maxs - mins)

    return (matrix - mins) / denom * 100


def cosine_similarity(target, matrix):
    target = np.asarray(target, dtype=float)
    matrix = np.asarray(matrix, dtype=float)

    target_norm = np.linalg.norm(target)
    matrix_norm = np.linalg.norm(matrix, axis=1)

    denom = matrix_norm * target_norm
    denom = np.where(denom == 0, 1e-9, denom)

    return matrix.dot(target) / denom


def get_recommendation_df(target_vector, top_n=5, exclude_song_id=None, use_rescale=True):
    df = songs.copy()
    matrix = get_vector(df)

    target = np.asarray(target_vector, dtype=float)

    if use_rescale:
        all_vectors = np.vstack([matrix, target])
        rescaled = minmax_rescale_matrix(all_vectors)
        matrix = rescaled[:-1]
        target = rescaled[-1]

    df["similarity"] = cosine_similarity(target, matrix)

    if exclude_song_id is not None:
        df = df[df["song_id"] != exclude_song_id]

    return df.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)


def tags_to_mood_vector(tags):
    scores = {col: 50 for col in MOOD_COLS}

    for tag in tags:
        if tag in TAG_MOOD_RULES:
            for col in MOOD_COLS:
                scores[col] += TAG_MOOD_RULES[tag].get(col, 0)

    for col in MOOD_COLS:
        scores[col] = max(0, min(100, scores[col]))

    return np.array([scores[col] for col in MOOD_COLS], dtype=float)


def tokenize_text(text):
    text = str(text).lower()
    words = re.findall(r"[가-힣a-zA-Z0-9]+", text)

    joined = re.sub(r"\s+", "", text)
    joined = re.sub(r"[^가-힣a-zA-Z0-9]", "", joined)

    ngrams = []

    for n in [2, 3]:
        for i in range(max(0, len(joined) - n + 1)):
            ngrams.append(joined[i:i+n])

    return words + ngrams


def vectorize_tokens(tokens, idf):
    counter = Counter(tokens)
    vec = {}

    for token, count in counter.items():
        vec[token] = count * idf.get(token, 1.0)

    return vec


def cosine_dict(vec1, vec2):
    if not vec1 or not vec2:
        return 0.0

    common = set(vec1.keys()).intersection(vec2.keys())
    dot = sum(vec1[k] * vec2[k] for k in common)

    norm1 = np.sqrt(sum(v * v for v in vec1.values()))
    norm2 = np.sqrt(sum(v * v for v in vec2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot / (norm1 * norm2))


@st.cache_data
def build_local_tag_vectors():
    docs = {}
    all_doc_tokens = []

    for tag, desc in TAG_DESCRIPTIONS.items():
        keyword_text = " ".join(TEXT_KEYWORDS.get(tag, []))
        full_text = f"{tag} {desc} {keyword_text}"
        tokens = tokenize_text(full_text)
        docs[tag] = tokens
        all_doc_tokens.append(set(tokens))

    doc_count = len(all_doc_tokens)
    df_counter = Counter()

    for token_set in all_doc_tokens:
        for token in token_set:
            df_counter[token] += 1

    idf = {}

    for token, df_count in df_counter.items():
        idf[token] = np.log((doc_count + 1) / (df_count + 1)) + 1

    tag_vectors = {}

    for tag, tokens in docs.items():
        tag_vectors[tag] = vectorize_tokens(tokens, idf)

    return tag_vectors, idf


def analyze_text_local(text):
    text = str(text).strip()

    if not text:
        return {
            "source": "local",
            "detected_tags": [],
            "scores": None,
            "reason": "입력 문장이 비어 있습니다."
        }

    tag_vectors, idf = build_local_tag_vectors()
    user_tokens = tokenize_text(text)
    user_vector = vectorize_tokens(user_tokens, idf)

    scores = []

    for tag, tag_vec in tag_vectors.items():
        sim = cosine_dict(user_vector, tag_vec)

        keyword_bonus = 0.0
        text_lower = text.lower()

        for keyword in TEXT_KEYWORDS.get(tag, []):
            if keyword.lower() in text_lower:
                keyword_bonus += 0.08

        final_score = sim + keyword_bonus
        scores.append((tag, final_score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    max_score = scores[0][1] if scores else 0

    if max_score <= 0:
        detected_tags = ["dreamy", "calm", "lonely"]
    else:
        detected_tags = []

        for tag, score in scores:
            if len(detected_tags) >= 5:
                break

            if score >= max_score * 0.45 or len(detected_tags) < 3:
                detected_tags.append(tag)

    mood_vector = tags_to_mood_vector(detected_tags)
    mood_scores = {col: float(mood_vector[i]) for i, col in enumerate(MOOD_COLS)}

    top_desc = ", ".join(detected_tags)

    return {
        "source": "local",
        "detected_tags": detected_tags,
        "scores": mood_scores,
        "reason": f"입력 문장과 분위기 태그 설명문을 비교하여 {top_desc} 태그를 감지했습니다."
    }


def build_target_vector_from_analysis(analysis, manual_tags):
    final_tags = []

    for tag in analysis.get("detected_tags", []) + list(manual_tags):
        if tag in TAG_MOOD_RULES and tag not in final_tags:
            final_tags.append(tag)

    if analysis.get("scores"):
        vector = np.array([analysis["scores"].get(col, 50) for col in MOOD_COLS], dtype=float)

        if manual_tags:
            manual_vector = tags_to_mood_vector(manual_tags)
            vector = vector * 0.7 + manual_vector * 0.3

        return vector, final_tags

    if final_tags:
        return tags_to_mood_vector(final_tags), final_tags

    return np.array([50 for _ in MOOD_COLS], dtype=float), final_tags


def mood_bar_chart(row_or_vector, title="분위기 스탯"):
    if isinstance(row_or_vector, pd.Series):
        values = [float(row_or_vector[col]) for col in MOOD_COLS]
    else:
        values = list(row_or_vector)

    chart_df = pd.DataFrame({
        "분위기": [MOOD_KO[col] for col in MOOD_COLS],
        "점수": values
    })

    fig = px.bar(
        chart_df,
        x="분위기",
        y="점수",
        text="점수",
        title=title,
        range_y=[0, 100]
    )

    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=55, b=10),
        yaxis_title="점수",
        xaxis_title=None
    )

    st.plotly_chart(fig, use_container_width=True)


def mood_radar_chart(values, title="분위기 프로필"):
    if isinstance(values, pd.Series):
        scores = [float(values[col]) for col in MOOD_COLS]
    else:
        scores = list(values)

    labels = [MOOD_KO[col] for col in MOOD_COLS]
    scores_closed = scores + [scores[0]]
    labels_closed = labels + [labels[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=scores_closed,
            theta=labels_closed,
            fill="toself",
            name="분위기"
        )
    )

    fig.update_layout(
        title=title,
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=420,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)


def render_song_card(row, rank=None, show_similarity=False):
    title_ko = row.get("title_ko", "")
    title_original = row.get("title_original", "")
    producers = join_display(row.get("producers", ""))
    engines = join_display(row.get("vocal_engines", ""))
    tags = split_values(row.get("mood_tags", ""))
    link = row.get("main_link", "")

    prefix = f"{rank}. " if rank is not None else ""

    st.markdown('<div class="song-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="song-title">{prefix}{title_ko} <span class="small-muted">({title_original})</span></div>',
        unsafe_allow_html=True
    )
    st.markdown(f"**보카로P:** {producers}")
    st.markdown(f"**음성합성엔진:** {engines}")

    if show_similarity and "similarity" in row:
        st.markdown(f"**유사도:** {float(row['similarity']) * 100:.1f}%")

    if tags:
        tag_html = " ".join([f'<span class="tag-pill">{tag}</span>' for tag in tags])
        st.markdown(tag_html, unsafe_allow_html=True)

    if isinstance(link, str) and link.strip():
        st.markdown(f"[YouTube에서 열기]({link})")

    st.markdown("</div>", unsafe_allow_html=True)


def get_same_producer_songs(selected_row, limit=8):
    selected_id = selected_row["song_id"]
    selected_producers = set(split_values(selected_row["producers"]))

    if not selected_producers:
        return songs.iloc[0:0]

    result = songs[
        songs["producers"].apply(lambda x: bool(selected_producers.intersection(set(split_values(x)))))
        & (songs["song_id"] != selected_id)
    ].copy()

    return result.head(limit)


def get_same_engine_songs(selected_row, limit=8):
    selected_id = selected_row["song_id"]
    selected_engines = set(split_values(selected_row["vocal_engines"]))

    if not selected_engines:
        return songs.iloc[0:0]

    result = songs[
        songs["vocal_engines"].apply(lambda x: bool(selected_engines.intersection(set(split_values(x)))))
        & (songs["song_id"] != selected_id)
    ].copy()

    return result.head(limit)


def extract_youtube_id(url):
    if not isinstance(url, str):
        return None

    url = url.strip()

    if not url:
        return None

    try:
        parsed = urlparse(url)

        if parsed.netloc in ["youtu.be", "www.youtu.be"]:
            video_id = parsed.path.strip("/").split("/")[0]
            return video_id or None

        if "youtube.com" in parsed.netloc:
            if parsed.path.startswith("/watch"):
                query = parse_qs(parsed.query)
                return query.get("v", [None])[0]

            if parsed.path.startswith("/embed/"):
                return parsed.path.split("/embed/")[1].split("/")[0]

            if parsed.path.startswith("/shorts/"):
                return parsed.path.split("/shorts/")[1].split("/")[0]

    except Exception:
        return None

    return None


def youtube_player(link):
    video_id = extract_youtube_id(link)

    if video_id:
        embed_url = f"https://www.youtube.com/embed/{video_id}"
        components.iframe(embed_url, height=420)
    elif isinstance(link, str) and link.strip():
        st.video(link)
    else:
        st.info("영상 링크가 없습니다.")


def normalized_entropy(counts):
    values = np.array(counts, dtype=float)

    if len(values) <= 1 or values.sum() == 0:
        return 0

    p = values / values.sum()
    p = p[p > 0]

    return float(-(p * np.log(p)).sum() / np.log(len(values)))


def top_share(counts):
    values = np.array(counts, dtype=float)

    if len(values) == 0 or values.sum() == 0:
        return 0

    return float(values.max() / values.sum())


def explode_items(df, column, item_name):
    records = []

    for _, row in df.iterrows():
        for item in split_values(row.get(column, "")):
            record = {
                "song_id": row["song_id"],
                item_name: item
            }

            for col in MOOD_COLS:
                record[col] = row[col]

            record["title_ko"] = row.get("title_ko", "")
            record["title_original"] = row.get("title_original", "")
            record["producers"] = row.get("producers", "")
            record["vocal_engines"] = row.get("vocal_engines", "")
            record["mood_tags"] = row.get("mood_tags", "")
            records.append(record)

    return pd.DataFrame(records)


def pca_2d(matrix):
    X = np.asarray(matrix, dtype=float)
    X = X - X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1, std)
    X = X / std

    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    coords = X @ Vt[:2].T

    return coords


def dominant_mood_from_row(row):
    values = row[MOOD_COLS].astype(float)
    col = values.idxmax()
    return MOOD_KO[col]


def set_emotion_example(text):
    st.session_state["emotion_input"] = text


def group_profile(df, source_column, group_name):
    exploded = explode_items(df, source_column, group_name)

    if exploded.empty:
        return pd.DataFrame()

    profile = exploded.groupby(group_name).agg(
        song_count=("song_id", "nunique"),
        joy=("joy", "mean"),
        sadness=("sadness", "mean"),
        dreamy=("dreamy", "mean"),
        tension=("tension", "mean"),
        dark=("dark", "mean"),
        energy=("energy", "mean"),
        cute=("cute", "mean"),
        chaotic=("chaotic", "mean")
    ).reset_index()

    profile["dominant_mood_key"] = profile[MOOD_COLS].idxmax(axis=1)
    profile["대표 분위기"] = profile["dominant_mood_key"].map(MOOD_KO)
    profile["대표 분위기 점수"] = profile[MOOD_COLS].max(axis=1).round(1)

    return profile


def make_one_axis_position_plot(profile, label_col, mood_col, title):
    if profile.empty:
        st.info("표시할 데이터가 부족합니다.")
        return

    plot_df = profile.copy()
    plot_df = plot_df.sort_values(mood_col, ascending=True).reset_index(drop=True)
    plot_df["score"] = plot_df[mood_col].round(1)
    plot_df["label"] = plot_df[label_col].astype(str)

    jitter_pattern = [-0.18, 0.18, -0.09, 0.09, 0]
    plot_df["line_y"] = [jitter_pattern[i % len(jitter_pattern)] for i in range(len(plot_df))]

    fig = go.Figure()

    fig.add_shape(
        type="line",
        x0=0,
        x1=100,
        y0=0,
        y1=0,
        line=dict(width=3)
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df["score"],
            y=plot_df["line_y"],
            mode="markers+text",
            text=plot_df["label"],
            textposition="top center",
            marker=dict(
                size=np.clip(plot_df["song_count"] * 4 + 10, 12, 42),
                opacity=0.82
            ),
            customdata=np.stack(
                [
                    plot_df["song_count"],
                    plot_df["대표 분위기"],
                    plot_df["대표 분위기 점수"]
                ],
                axis=-1
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                + f"{MOOD_KO[mood_col]} 점수: "
                + "%{x:.1f}<br>"
                + "곡 수: %{customdata[0]}<br>"
                + "대표 분위기: %{customdata[1]}<br>"
                + "대표 분위기 점수: %{customdata[2]:.1f}"
                + "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=title,
        height=540,
        xaxis=dict(
            title=f"{MOOD_KO[mood_col]} 점수",
            range=[0, 100],
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            visible=False,
            range=[-0.55, 0.55]
        ),
        margin=dict(l=10, r=10, t=70, b=40),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


def build_profile_table(profile, label_col):
    if profile.empty:
        return pd.DataFrame()

    table = profile.copy()
    table = table.sort_values(["song_count", "대표 분위기 점수"], ascending=[False, False])

    rename_map = {
        label_col: "이름",
        "song_count": "곡 수",
        "대표 분위기": "대표 분위기",
        "대표 분위기 점수": "대표 분위기 점수",
        **MOOD_KO
    }

    table = table.rename(columns=rename_map)

    ordered_cols = [
        "이름", "곡 수", "대표 분위기", "대표 분위기 점수",
        "기쁨", "슬픔", "몽환", "긴장감",
        "어두움", "에너지", "귀여움", "혼란스러움"
    ]

    existing_cols = [col for col in ordered_cols if col in table.columns]
    table = table[existing_cols]

    score_cols = [
        "대표 분위기 점수", "기쁨", "슬픔", "몽환", "긴장감",
        "어두움", "에너지", "귀여움", "혼란스러움"
    ]

    for col in score_cols:
        if col in table.columns:
            table[col] = table[col].round(1)

    return table


def engine_combination_tables(df):
    exact_counter = Counter()
    pair_counter = Counter()
    examples = {}

    for _, row in df.iterrows():
        engines = sorted(set(split_values(row.get("vocal_engines", ""))))

        if len(engines) >= 2:
            combo_name = " + ".join(engines)
            exact_counter[combo_name] += 1

            if combo_name not in examples:
                examples[combo_name] = []

            examples[combo_name].append(str(row.get("title_ko", "")))

            for pair in combinations(engines, 2):
                pair_name = " + ".join(pair)
                pair_counter[pair_name] += 1

    exact_df = pd.DataFrame(
        [
            {
                "조합": combo,
                "곡 수": count,
                "예시 곡": ", ".join(examples.get(combo, [])[:4])
            }
            for combo, count in exact_counter.items()
        ]
    )

    pair_df = pd.DataFrame(
        [
            {
                "엔진쌍": pair,
                "함께 등장한 곡 수": count
            }
            for pair, count in pair_counter.items()
        ]
    )

    if not exact_df.empty:
        exact_df = exact_df.sort_values("곡 수", ascending=False).reset_index(drop=True)

    if not pair_df.empty:
        pair_df = pair_df.sort_values("함께 등장한 곡 수", ascending=False).reset_index(drop=True)

    return exact_df, pair_df


def get_song_by_id(song_id):
    matched = songs[songs["song_id"].astype(str) == str(song_id)]

    if matched.empty:
        return None

    return matched.iloc[0]


# =========================================================
# 사이드바 네비게이션
# =========================================================

MENU = ["Home", "감정 기반 추천", "유사 곡 탐색", "랜덤 곡 추천", "트렌드 분석", "DB 확인"]

if "sidebar_page" not in st.session_state:
    st.session_state["sidebar_page"] = "Home"

if st.session_state["sidebar_page"] not in MENU:
    st.session_state["sidebar_page"] = "Home"

if "pending_page" in st.session_state:
    pending_page = st.session_state.pop("pending_page")

    if pending_page in MENU:
        st.session_state["sidebar_page"] = pending_page

st.sidebar.title("🎧 Vocaloid Recommender")

page = st.sidebar.radio(
    "메뉴",
    MENU,
    key="sidebar_page"
)

st.sidebar.markdown("---")


# =========================================================
# Home
# =========================================================

if page == "Home":
    st.markdown('<div class="main-title">보컬로이드 감성 분석 및 노래 추천 웹앱</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-text">사용자의 현재 감정과 원하는 분위기를 바탕으로 보컬로이드 곡을 추천하고, 비슷한 분위기의 곡을 탐색하는 웹앱입니다.</div>',
        unsafe_allow_html=True
    )

    producer_count = len(count_split_column(songs, "producers"))
    engine_count = len(count_split_column(songs, "vocal_engines"))
    tag_count = len(count_split_column(songs, "mood_tags"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("수록 곡 수", f"{len(songs)}곡")
    c2.metric("보카로P 수", f"{producer_count}명")
    c3.metric("음성합성엔진 수", f"{engine_count}종")
    c4.metric("분위기 태그 수", f"{tag_count}개")

    st.markdown("### 주요 기능")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("#### 1. 감정 기반 추천")
        st.write("자연어 입력을 로컬 분석기로 분위기 벡터화한 뒤 곡을 추천합니다.")

    with col2:
        st.markdown("#### 2. 유사 곡 탐색")
        st.write("선택한 곡과 분위기가 비슷한 곡, 같은 보카로P 곡, 같은 엔진 곡을 보여줍니다.")

    with col3:
        st.markdown("#### 3. 랜덤 곡 추천")
        st.write("버튼 하나로 랜덤 곡을 추천받고, 바로 유사 곡 탐색으로 넘어갈 수 있습니다.")

    with col4:
        st.markdown("#### 4. 트렌드 분석")
        st.write("DB 내부의 태그, 프로듀서, 엔진, 분위기 구조를 그래프와 해석으로 분석합니다.")

    st.markdown("### 현재 DB 분위기 요약")
    avg_mood = songs[MOOD_COLS].mean()
    mood_radar_chart(avg_mood, title="전체 수록곡 평균 분위기 프로필")

    st.markdown("### 곡 미리보기")
    preview_df = songs[["song_id", "title_ko", "title_original", "producers", "vocal_engines", "mood_tags"]].head(12)
    st.dataframe(preview_df, use_container_width=True)


# =========================================================
# 감정 기반 추천
# =========================================================

elif page == "감정 기반 추천":
    st.markdown('<div class="main-title">감정 기반 노래 추천</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-text">자연어로 현재 기분을 입력하면 로컬 자연어 분석기가 분위기 벡터를 만들고, DB의 곡들과 비교해 추천합니다.</div>',
        unsafe_allow_html=True
    )

    if "emotion_input" not in st.session_state:
        st.session_state["emotion_input"] = ""

    emotion_text = st.text_area(
        "감정 또는 원하는 분위기를 자연어로 입력하세요",
        key="emotion_input",
        height=120,
        placeholder="예: 새벽에 멍하니 듣기 좋은, 너무 밝지는 않은데 어딘가 아련하고 중독성 있는 곡을 듣고 싶어"
    )

    st.markdown("### 예시 입력")
    b1, b2, b3, b4, b5 = st.columns(5)

    with b1:
        st.button(
            "우울하고 몽환적인 곡",
            use_container_width=True,
            on_click=set_emotion_example,
            args=("요즘 좀 우울해서 몽환적이고 쓸쓸한 곡을 듣고 싶어",)
        )

    with b2:
        st.button(
            "밝고 귀여운 곡",
            use_container_width=True,
            on_click=set_emotion_example,
            args=("가볍고 밝고 귀여운 분위기의 노래가 듣고 싶어",)
        )

    with b3:
        st.button(
            "강렬하고 혼란스러운 곡",
            use_container_width=True,
            on_click=set_emotion_example,
            args=("머리가 터질 것처럼 강렬하고 정신없는 곡이 듣고 싶어",)
        )

    with b4:
        st.button(
            "차분하고 아련한 곡",
            use_container_width=True,
            on_click=set_emotion_example,
            args=("밤에 조용히 듣기 좋은 차분하고 아련한 곡이 듣고 싶어",)
        )

    with b5:
        st.button(
            "어둡고 긴장감 있는 곡",
            use_container_width=True,
            on_click=set_emotion_example,
            args=("불안하고 어둡고 긴장감 있는 곡이 듣고 싶어",)
        )

    with st.expander("고급 설정 / 수동 보정"):
        available_tags = sorted(TAG_MOOD_RULES.keys())

        selected_tags = st.multiselect(
            "분석 결과에 추가할 분위기 태그",
            available_tags,
            help="분석 결과가 부족할 때 수동 태그로 추천 방향을 보정할 수 있습니다."
        )

        top_n = st.slider("추천 곡 수", 5, 15, 5)
        use_rescale = st.checkbox("추천 계산 시 분위기 스탯 리스케일 적용", value=True)

    if st.button("추천 받기", type="primary", use_container_width=True):
        if not emotion_text.strip():
            st.warning("감정이나 원하는 분위기를 입력해 주세요.")
        else:
            analysis = analyze_text_local(emotion_text)
            target_vector, final_tags = build_target_vector_from_analysis(analysis, selected_tags)

            recs = get_recommendation_df(
                target_vector,
                top_n=top_n,
                exclude_song_id=None,
                use_rescale=use_rescale
            )

            st.session_state["emotion_analysis"] = analysis
            st.session_state["emotion_final_tags"] = final_tags
            st.session_state["emotion_target_vector"] = target_vector.tolist()
            st.session_state["emotion_recs"] = recs.to_dict("records")
            st.session_state["emotion_selected_rank"] = 0

    if "emotion_recs" in st.session_state:
        analysis = st.session_state.get("emotion_analysis", {})
        final_tags = st.session_state.get("emotion_final_tags", [])
        target_vector = np.array(st.session_state.get("emotion_target_vector", [50] * len(MOOD_COLS)), dtype=float)
        recs_df = pd.DataFrame(st.session_state["emotion_recs"])

        st.markdown("---")
        st.markdown("### 자연어 분석 결과")

        a1, a2 = st.columns([1.3, 2.0])

        with a1:
            st.write("**감지된 태그:**")
            if final_tags:
                st.write(", ".join(final_tags))
            else:
                st.write("-")

            st.write("**분석 설명:**")
            st.info(analysis.get("reason", ""))

        with a2:
            mood_bar_chart(target_vector, title="입력 감정의 분위기 벡터")

        st.markdown("### 추천 결과")

        if recs_df.empty:
            st.info("추천 결과가 없습니다.")
        else:
            left_col, right_col = st.columns([1.0, 2.8])

            with left_col:
                st.markdown("#### 추천곡")
                st.markdown(
                    '<div class="rank-button-note">버튼을 누르면 오른쪽 영상과 정보가 바뀝니다.</div>',
                    unsafe_allow_html=True
                )

                for i, row in recs_df.iterrows():
                    button_label = f"{i + 1}. {row.get('title_ko', '')}"
                    button_type = "primary" if i == st.session_state.get("emotion_selected_rank", 0) else "secondary"

                    if st.button(
                        button_label,
                        key=f"emotion_rec_button_{i}_{row.get('song_id')}",
                        use_container_width=True,
                        type=button_type
                    ):
                        st.session_state["emotion_selected_rank"] = i
                        st.rerun()

            with right_col:
                selected_idx = st.session_state.get("emotion_selected_rank", 0)
                selected_idx = max(0, min(selected_idx, len(recs_df) - 1))
                selected_row = recs_df.iloc[selected_idx]

                st.markdown(f"#### 추천곡 {selected_idx + 1}: {selected_row.get('title_ko', '')}")
                youtube_player(selected_row.get("main_link", ""))

                st.markdown("#### 곡 정보")
                render_song_card(selected_row, rank=selected_idx + 1, show_similarity=True)

                info1, info2 = st.columns([1.0, 1.0])

                with info1:
                    mood_bar_chart(selected_row, title="선택 곡 분위기 스탯")

                with info2:
                    mood_radar_chart(selected_row, title="선택 곡 분위기 프로필")


# =========================================================
# 유사 곡 탐색
# =========================================================

elif page == "유사 곡 탐색":
    st.markdown('<div class="main-title">유사 곡 탐색</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-text">특정 곡을 선택하면 분위기 유사 곡, 같은 보카로P 곡, 같은 음성합성엔진 곡을 탐색합니다.</div>',
        unsafe_allow_html=True
    )

    song_options = []

    for _, row in songs.iterrows():
        label = f"{row['song_id']} | {row['title_ko']} ({row['title_original']})"
        song_options.append((label, row["song_id"]))

    option_map = {value: label for label, value in song_options}
    song_id_list = [x[1] for x in song_options]

    default_song_id = st.session_state.get("similar_target_song_id", song_id_list[0])

    if default_song_id not in song_id_list:
        default_song_id = song_id_list[0]

    default_index = song_id_list.index(default_song_id)

    selected_song_id = st.selectbox(
        "곡을 선택하세요",
        options=song_id_list,
        index=default_index,
        format_func=lambda sid: option_map.get(sid, sid)
    )

    st.session_state["similar_target_song_id"] = selected_song_id

    selected_row = songs[songs["song_id"] == selected_song_id].iloc[0]

    st.markdown("### 선택한 곡")
    top_left, top_right = st.columns([1.2, 1.5])

    with top_left:
        render_song_card(selected_row)

    with top_right:
        youtube_player(selected_row.get("main_link", ""))

    st.markdown("### 선택한 곡의 분위기")
    c1, c2 = st.columns(2)

    with c1:
        mood_bar_chart(selected_row, title="분위기 스탯")

    with c2:
        mood_radar_chart(selected_row, title="분위기 프로필")

    use_rescale = st.checkbox("유사도 계산 시 분위기 스탯 리스케일 적용", value=True)
    top_n = st.slider("유사 곡 추천 수", 3, 15, 5)

    target_vector = selected_row[MOOD_COLS].astype(float).to_numpy()

    similar_df = get_recommendation_df(
        target_vector,
        top_n=top_n,
        exclude_song_id=selected_song_id,
        use_rescale=use_rescale
    )

    tab1, tab2, tab3= st.tabs([
        "분위기 유사 곡",
        "같은 보카로P 곡",
        "같은 음성합성엔진 곡"
    ])

    with tab1:
        st.markdown("#### 분위기 스탯이 비슷한 곡")

        for idx, (_, row) in enumerate(similar_df.iterrows(), start=1):
            render_song_card(row, rank=idx, show_similarity=True)

    with tab2:
        st.markdown("#### 같은 보카로P가 참여한 곡")
        same_p = get_same_producer_songs(selected_row, limit=10)

        if same_p.empty:
            st.info("같은 보카로P의 다른 곡이 없습니다.")
        else:
            for idx, (_, row) in enumerate(same_p.iterrows(), start=1):
                render_song_card(row, rank=idx)

    with tab3:
        st.markdown("#### 같은 음성합성엔진을 사용한 곡")
        same_engine = get_same_engine_songs(selected_row, limit=10)

        if same_engine.empty:
            st.info("같은 음성합성엔진을 사용한 다른 곡이 없습니다.")
        else:
            for idx, (_, row) in enumerate(same_engine.iterrows(), start=1):
                render_song_card(row, rank=idx)

# =========================================================
# 랜덤 곡 추천
# =========================================================

elif page == "랜덤 곡 추천":
    st.markdown('<div class="main-title">랜덤 곡 추천</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-text">버튼을 누르면 DB에서 무작위로 보컬로이드 곡을 하나 추천합니다.</div>',
        unsafe_allow_html=True
    )

    left_space, center_col, right_space = st.columns([1, 1.2, 1])

    with center_col:
        if st.button("🎲 랜덤 곡 추천받기", use_container_width=True, type="primary"):
            random_row = songs.sample(1).iloc[0]
            st.session_state["random_song_id"] = random_row["song_id"]

    random_song_id = st.session_state.get("random_song_id", None)

    if random_song_id is None:
        st.info("아직 추천된 곡이 없습니다. 위 버튼을 눌러 랜덤 곡을 추천받아보세요.")
    else:
        random_row = get_song_by_id(random_song_id)

        if random_row is None:
            st.warning("저장된 랜덤 곡을 찾을 수 없습니다. 다시 추천받아 주세요.")
        else:
            st.markdown("---")
            st.markdown(f"### 오늘의 랜덤 추천곡: {random_row.get('title_ko', '')}")

            video_col, info_col = st.columns([1.5, 1.1])

            with video_col:
                youtube_player(random_row.get("main_link", ""))

            with info_col:
                render_song_card(random_row)

            st.markdown("### 분위기 정보")
            m1, m2 = st.columns(2)

            with m1:
                mood_bar_chart(random_row, title="랜덤 추천곡 분위기 스탯")

            with m2:
                mood_radar_chart(random_row, title="랜덤 추천곡 분위기 프로필")

            st.markdown("---")

            link_col1, link_col2, link_col3 = st.columns([1, 1.2, 1])

            with link_col2:
                if st.button("이 곡과 유사한 곡 추천받기", use_container_width=True, type="primary"):
                    st.session_state["similar_target_song_id"] = random_row["song_id"]
                    st.session_state["pending_page"] = "유사 곡 탐색"
                    st.rerun()


# =========================================================
# 트렌드 분석
# =========================================================

elif page == "트렌드 분석":
    st.markdown('<div class="main-title">트렌드 및 DB 분석</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-text">현재 DB의 전체 분위기, 집중도, 보카로P 위치, 음성합성엔진 위치, 엔진 조합을 직관적으로 분석합니다.</div>',
        unsafe_allow_html=True
    )

    tag_counts = count_split_column(songs, "mood_tags")
    producer_counts = count_split_column(songs, "producers")
    engine_counts = count_split_column(songs, "vocal_engines")

    tag_entropy = normalized_entropy(tag_counts["count"]) if not tag_counts.empty else 0
    producer_entropy = normalized_entropy(producer_counts["count"]) if not producer_counts.empty else 0
    engine_entropy = normalized_entropy(engine_counts["count"]) if not engine_counts.empty else 0

    tag_top_share = top_share(tag_counts["count"]) if not tag_counts.empty else 0
    producer_top_share = top_share(producer_counts["count"]) if not producer_counts.empty else 0
    engine_top_share = top_share(engine_counts["count"]) if not engine_counts.empty else 0

    top_tag_name = tag_counts.iloc[0]["name"] if not tag_counts.empty else "-"
    top_producer_name = producer_counts.iloc[0]["name"] if not producer_counts.empty else "-"
    top_engine_name = engine_counts.iloc[0]["name"] if not engine_counts.empty else "-"

    avg_mood = songs[MOOD_COLS].mean()
    dominant_col = avg_mood.idxmax()
    weakest_col = avg_mood.idxmin()

    multi_engine_count = songs["vocal_engines"].apply(lambda x: len(split_values(x)) >= 2).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("수록 곡 수", f"{len(songs)}곡")
    c2.metric("가장 강한 평균 분위기", MOOD_KO[dominant_col], f"{avg_mood[dominant_col]:.1f}")
    c3.metric("다중 엔진 곡", f"{multi_engine_count}곡")
    c4.metric("프로듀서 최상위 집중도", f"{producer_top_share * 100:.1f}%")

    trend_tabs = st.tabs([
        "요약",
        "곡 분위기 지도",
        "보카로P 위치",
        "음성합성엔진 위치",
        "엔진 조합 분석"
    ])

    with trend_tabs[0]:
        st.markdown("### 전체 DB 요약")

        left, right = st.columns([1.1, 1.1])

        with left:
            mood_radar_chart(avg_mood, title="전체 평균 분위기 프로필")

        with right:
            mood_bar_chart(avg_mood, title="전체 평균 분위기 점수")

        st.markdown(
            f"""
            <div class="analysis-box">
            <b>요약 해석</b><br>
            현재 DB의 평균 중심축은 <b>{MOOD_KO[dominant_col]}</b>입니다.
            반대로 가장 약한 축은 <b>{MOOD_KO[weakest_col]}</b>입니다.
            따라서 이 DB는 전체적으로 <b>{MOOD_KO[dominant_col]}</b> 성향이 상대적으로 강한 추천 데이터셋으로 볼 수 있습니다.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 집중도 및 다양성 지표")

        k1, k2, k3 = st.columns(3)

        with k1:
            st.metric("태그 최상위 집중도", f"{tag_top_share * 100:.1f}%")
            st.caption(f"최다 태그: {top_tag_name}")

        with k2:
            st.metric("보카로P 최상위 집중도", f"{producer_top_share * 100:.1f}%")
            st.caption(f"최다 보카로P: {top_producer_name}")

        with k3:
            st.metric("엔진 최상위 집중도", f"{engine_top_share * 100:.1f}%")
            st.caption(f"최다 엔진: {top_engine_name}")

        d1, d2, d3 = st.columns(3)

        with d1:
            st.metric("태그 다양성 지수", f"{tag_entropy:.2f}")
            st.caption("1에 가까울수록 태그가 고르게 분산되어 있습니다.")

        with d2:
            st.metric("보카로P 다양성 지수", f"{producer_entropy:.2f}")
            st.caption("1에 가까울수록 특정 P에 덜 몰려 있습니다.")

        with d3:
            st.metric("엔진 다양성 지수", f"{engine_entropy:.2f}")
            st.caption("1에 가까울수록 특정 엔진에 덜 몰려 있습니다.")

        st.markdown(
            f"""
            <div class="analysis-box">
            <b>집중도 해석</b><br>
            최상위 보카로P <b>{top_producer_name}</b>의 비중은 <b>{producer_top_share * 100:.1f}%</b>이고,
            최상위 엔진 <b>{top_engine_name}</b>의 비중은 <b>{engine_top_share * 100:.1f}%</b>입니다.
            보컬로이드 씬은 특정 인기 P와 대표 음성합성엔진에 소비가 몰리는 경향이 있으므로,
            집중도는 오류라기보다 현재 DB가 어떤 취향권과 중심축을 갖는지 보여주는 지표입니다.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 상위 태그 요약표")
        tag_exploded = explode_items(songs, "mood_tags", "tag")

        if tag_exploded.empty:
            st.info("태그 데이터가 부족합니다.")
        else:
            tag_profile = tag_exploded.groupby("tag").agg(
                곡수=("song_id", "nunique"),
                joy=("joy", "mean"),
                sadness=("sadness", "mean"),
                dreamy=("dreamy", "mean"),
                tension=("tension", "mean"),
                dark=("dark", "mean"),
                energy=("energy", "mean"),
                cute=("cute", "mean"),
                chaotic=("chaotic", "mean")
            ).reset_index()

            tag_profile["대표 분위기"] = tag_profile[MOOD_COLS].idxmax(axis=1).map(MOOD_KO)
            tag_profile["대표 분위기 점수"] = tag_profile[MOOD_COLS].max(axis=1).round(1)
            tag_profile = tag_profile.sort_values("곡수", ascending=False)

            tag_table = tag_profile[["tag", "곡수", "대표 분위기", "대표 분위기 점수"]].rename(
                columns={"tag": "태그"}
            ).head(20)

            st.dataframe(tag_table, use_container_width=True, hide_index=True)

        st.markdown("### 상위 보카로P / 엔진")
        p_col, e_col = st.columns(2)

        with p_col:
            if producer_counts.empty:
                st.info("보카로P 데이터가 없습니다.")
            else:
                fig = px.bar(
                    producer_counts.head(10).sort_values("count", ascending=True),
                    x="count",
                    y="name",
                    orientation="h",
                    title="상위 보카로P"
                )
                fig.update_layout(height=420, xaxis_title="등장 곡 수", yaxis_title=None)
                st.plotly_chart(fig, use_container_width=True)

        with e_col:
            if engine_counts.empty:
                st.info("엔진 데이터가 없습니다.")
            else:
                fig = px.bar(
                    engine_counts.head(10).sort_values("count", ascending=True),
                    x="count",
                    y="name",
                    orientation="h",
                    title="상위 음성합성엔진"
                )
                fig.update_layout(height=420, xaxis_title="등장 곡 수", yaxis_title=None)
                st.plotly_chart(fig, use_container_width=True)

    with trend_tabs[1]:
        st.markdown("### 곡들의 분위기 분포 지도")

        coords = pca_2d(songs[MOOD_COLS].to_numpy())
        scatter_df = songs.copy()
        scatter_df["PC1"] = coords[:, 0]
        scatter_df["PC2"] = coords[:, 1]
        scatter_df["대표 분위기"] = scatter_df.apply(dominant_mood_from_row, axis=1)
        scatter_df["표시명"] = scatter_df["title_ko"].astype(str) + " / " + scatter_df["title_original"].astype(str)

        fig = px.scatter(
            scatter_df,
            x="PC1",
            y="PC2",
            color="대표 분위기",
            hover_name="표시명",
            hover_data={
                "producers": True,
                "vocal_engines": True,
                "mood_tags": True,
                "PC1": False,
                "PC2": False
            },
            title="분위기 스탯 기반 곡 분포 지도"
        )

        fig.update_layout(height=620, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            <div class="analysis-box">
            <b>해석</b><br>
            이 지도는 8개의 분위기 스탯을 2차원으로 압축한 것입니다.
            가까운 곡들은 분위기 조합이 비슷하고, 멀리 떨어진 곡들은 분위기 구조가 다릅니다.
            감정 기반 추천과 유사 곡 추천은 이와 같은 벡터 유사도 개념을 사용합니다.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### 분위기 축 상관관계")
        corr = songs[MOOD_COLS].corr()
        corr = corr.rename(index=MOOD_KO, columns=MOOD_KO)

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="분위기 스탯 상관관계"
        )

        fig.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with trend_tabs[2]:
        st.markdown("### 보카로P는 하나의 분위기 축에서 어디에 위치하는가")

        producer_profile = group_profile(songs, "producers", "producer")

        if producer_profile.empty:
            st.info("보카로P 데이터가 부족합니다.")
        else:
            min_p_count = st.slider("표시할 보카로P 최소 곡 수", 1, 10, 3, key="producer_axis_min_count")
            producer_profile_filtered = producer_profile[producer_profile["song_count"] >= min_p_count].copy()

            axis_p = st.selectbox(
                "기준 분위기 축",
                MOOD_COLS,
                index=MOOD_COLS.index("energy"),
                format_func=lambda x: MOOD_KO[x],
                key="producer_single_axis"
            )

            if producer_profile_filtered.empty:
                st.info("조건에 맞는 보카로P가 없습니다. 최소 곡 수를 낮춰보세요.")
            else:
                make_one_axis_position_plot(
                    producer_profile_filtered,
                    "producer",
                    axis_p,
                    title=f"보카로P 위치: {MOOD_KO[axis_p]} 축"
                )

                st.markdown(
                    f"""
                    <div class="analysis-box">
                    <b>해석</b><br>
                    이 그래프는 보카로P들을 <b>{MOOD_KO[axis_p]}</b>이라는 하나의 분위기 축 위에 배치한 것입니다.
                    오른쪽으로 갈수록 해당 P의 곡들이 평균적으로 <b>{MOOD_KO[axis_p]}</b> 성향이 강합니다.
                    점의 크기는 DB에 포함된 곡 수를 의미합니다.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("### 보카로P 성향 요약표")
                producer_table = build_profile_table(producer_profile_filtered, "producer")
                st.dataframe(producer_table, use_container_width=True, hide_index=True)

    with trend_tabs[3]:
        st.markdown("### 음성합성엔진은 하나의 분위기 축에서 어디에 위치하는가")

        engine_profile = group_profile(songs, "vocal_engines", "engine")

        if engine_profile.empty:
            st.info("음성합성엔진 데이터가 부족합니다.")
        else:
            min_e_count = st.slider("표시할 엔진 최소 곡 수", 1, 10, 2, key="engine_axis_min_count")
            engine_profile_filtered = engine_profile[engine_profile["song_count"] >= min_e_count].copy()

            axis_e = st.selectbox(
                "기준 분위기 축",
                MOOD_COLS,
                index=MOOD_COLS.index("cute"),
                format_func=lambda x: MOOD_KO[x],
                key="engine_single_axis"
            )

            if engine_profile_filtered.empty:
                st.info("조건에 맞는 엔진이 없습니다. 최소 곡 수를 낮춰보세요.")
            else:
                make_one_axis_position_plot(
                    engine_profile_filtered,
                    "engine",
                    axis_e,
                    title=f"음성합성엔진 위치: {MOOD_KO[axis_e]} 축"
                )

                st.markdown(
                    f"""
                    <div class="analysis-box">
                    <b>해석</b><br>
                    이 그래프는 음성합성엔진들을 <b>{MOOD_KO[axis_e]}</b>이라는 하나의 분위기 축 위에 배치한 것입니다.
                    오른쪽으로 갈수록 해당 엔진이 사용된 곡들이 평균적으로 <b>{MOOD_KO[axis_e]}</b> 성향이 강합니다.
                    점의 크기는 해당 엔진이 등장한 곡 수를 의미합니다.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("### 음성합성엔진 성향 요약표")
                engine_table = build_profile_table(engine_profile_filtered, "engine")
                st.dataframe(engine_table, use_container_width=True, hide_index=True)

    with trend_tabs[4]:
        st.markdown("### 음성합성엔진 조합 분석")

        exact_combo_df, pair_combo_df = engine_combination_tables(songs)

        multi_engine_ratio = multi_engine_count / len(songs) if len(songs) > 0 else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("다중 엔진 곡 수", f"{multi_engine_count}곡")
        m2.metric("다중 엔진 비율", f"{multi_engine_ratio * 100:.1f}%")
        m3.metric("엔진 조합 종류", f"{len(exact_combo_df)}종")

        st.markdown(
            """
            <div class="analysis-box">
            <b>해석</b><br>
            이 분석은 하나의 곡에 여러 음성합성엔진이 함께 사용된 경우를 따로 모아,
            어떤 엔진 조합이 자주 쓰이는지 확인합니다.
            단일 엔진 곡이 많은 보컬로이드 DB에서 다중 엔진 조합은 협업성이나 캐릭터 조합의 경향을 보여주는 지표가 될 수 있습니다.
            </div>
            """,
            unsafe_allow_html=True
        )

        if exact_combo_df.empty:
            st.info("다중 엔진 곡이 없습니다.")
        else:
            st.markdown("### 정확한 엔진 조합 TOP")
            fig = px.bar(
                exact_combo_df.head(15).sort_values("곡 수", ascending=True),
                x="곡 수",
                y="조합",
                orientation="h",
                title="자주 등장한 음성합성엔진 조합"
            )
            fig.update_layout(height=520, xaxis_title="곡 수", yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(exact_combo_df, use_container_width=True, hide_index=True)

        if pair_combo_df.empty:
            st.info("엔진쌍 분석 데이터가 없습니다.")
        else:
            st.markdown("### 엔진쌍 공동 등장 TOP")
            fig = px.bar(
                pair_combo_df.head(15).sort_values("함께 등장한 곡 수", ascending=True),
                x="함께 등장한 곡 수",
                y="엔진쌍",
                orientation="h",
                title="자주 함께 등장한 엔진쌍"
            )
            fig.update_layout(height=520, xaxis_title="함께 등장한 곡 수", yaxis_title=None)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(pair_combo_df.head(30), use_container_width=True, hide_index=True)


# =========================================================
# DB 확인
# =========================================================

elif page == "DB 확인":
    st.markdown('<div class="main-title">DB 확인</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-text">현재 songs_input.csv가 제대로 읽히는지 확인하는 화면입니다.</div>',
        unsafe_allow_html=True
    )

    st.write(f"데이터 파일 경로: `{find_data_path()}`")
    st.write(f"총 {len(songs)}곡")

    search = st.text_input("곡 제목, 보카로P, 엔진, 태그 검색")

    view_df = songs.copy()

    if search.strip():
        keyword = search.strip().lower()
        mask = (
            view_df["title_original"].astype(str).str.lower().str.contains(keyword, na=False)
            | view_df["title_ko"].astype(str).str.lower().str.contains(keyword, na=False)
            | view_df["producers"].astype(str).str.lower().str.contains(keyword, na=False)
            | view_df["vocal_engines"].astype(str).str.lower().str.contains(keyword, na=False)
            | view_df["mood_tags"].astype(str).str.lower().str.contains(keyword, na=False)
        )
        view_df = view_df[mask]

    show_cols = [
        "song_id", "title_original", "title_ko",
        "producers", "vocal_engines",
        "main_link", "mood_tags"
    ] + MOOD_COLS

    st.dataframe(view_df[show_cols], use_container_width=True)

    st.download_button(
        "현재 DB CSV 다운로드",
        data=songs.to_csv(index=False, encoding="utf-8-sig"),
        file_name="songs_input_checked.csv",
        mime="text/csv"
    )
