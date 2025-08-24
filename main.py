import os
import json
from typing import List, Dict, Any

import streamlit as st
from openai import OpenAI

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="Student Writing Evaluator (LLM)", page_icon="📝", layout="wide")

# ----------------------------
# Sidebar: Settings
# ----------------------------
st.sidebar.title("⚙️ Settings")
api_key = (
    st.sidebar.text_input("OpenAI API Key", type="password")
    or os.getenv("OPENAI_API_KEY")
    or st.secrets.get("OPENAI_API_KEY", None)
)
model_name = st.sidebar.text_input("Model", value="gpt-4o-mini")

st.sidebar.markdown(
    """
    **How to use**
    1) Enter your API key.  
    2) Choose a topic and paste a student's writing.  
    3) Click **Evaluate**.
    """
)

# ----------------------------
# Topics
# ----------------------------
DEFAULT_TOPICS = [
    "Does technological development change society? / 기술 발전은 사회를 변화시키는가?",
    "Wild Robot: Technology & Nature Coexistence / 와일드 로봇: 기술과 자연의 공존",
    "Digital Citizenship & Online Etiquette / 디지털 시민성과 온라인 예절",
    "AI Ethics in School Life / 학교생활 속 AI 윤리",
]

st.title("📝 Student Writing Evaluator • Streamlit + OpenAI")

colA, colB = st.columns([2, 1])
with colA:
    topic = st.selectbox(
        "Select a topic (or choose 'Custom') / 주제를 선택하세요",
        options=["Custom / 사용자 정의"] + DEFAULT_TOPICS,
        index=1,
    )
    custom_topic = ""
    if topic.startswith("Custom"):
        custom_topic = st.text_input("Enter your custom topic / 사용자 정의 주제를 입력하세요")

with colB:
    lang = st.selectbox("Feedback language / 피드백 언어", ["Korean", "English", "Bilingual"])

essay = st.text_area(
    "Paste student's writing here / 학생 글을 붙여넣으세요",
    height=280,
    placeholder=(
        "예: 기술 발전은 사회를 어떻게 바꾸는가에 대한 나의 생각은…\n"
        "(철자·띄어쓰기·문장 오류가 있어도 괜찮아요. 모두 잡아 드립니다.)"
    ),
)

# ----------------------------
# Rubric (from user spec)
# ----------------------------
RUBRIC = [
    {
        "id": "understanding",
        "name": "1) 내용 이해 / Understanding the contents",
        "desc": "작품/주제의 핵심 내용, 등장인물의 말과 행동, 개념을 정확히 이해하고 반영했는가?",
    },
    {
        "id": "ideas_arguments",
        "name": "2) 아이디어·주장 / Ideas & Argumentation",
        "desc": "주장과 근거가 명확하고 통찰이 있는가? 사례·데이터·인용 등으로 뒷받침했는가?",
    },
    {
        "id": "organization",
        "name": "3) 구성·전개 / Composition & Organization",
        "desc": "서론-본론-결론의 구조, 문단 간 논리 흐름, 연결어 사용이 자연스러운가?",
    },
    {
        "id": "expression",
        "name": "4) 표현·문장 / Expression & Sentences",
        "desc": "어휘 선택, 맞춤법·띄어쓰기, 문장 다양성(단문/복문/병렬)과 명료성이 적절한가?",
    },
    {
        "id": "attitude_integrity",
        "name": "5) 태도·성실성 / Attitude & Integrity",
        "desc": "분량, 주관적 성찰의 깊이, 과제 지침 준수, 인용 표기 등을 성실히 수행했는가?",
    },
]

RUBRIC_SCALE = {
    5: "탁월/Excellent",
    4: "우수/Strong",
    3: "보통/Adequate",
    2: "미흡/Limited",
    1: "부족/Weak",
}

# ----------------------------
# Prompt builder
# ----------------------------
SYSTEM_PROMPT = (
    "You are a meticulous Korean middle-school writing evaluator and editor. "
    "Follow the rubric strictly and produce exhaustive, actionable feedback. "
    "When correcting grammar/spelling/spacing, list *every* error with 'Original → Amended' and a short reason. "
)

USER_INSTRUCTIONS_TEMPLATE = """
You will evaluate a student's essay according to the following rubric and output **valid JSON only**.

Topic: "{topic}"

Rubric (5-point scale per item):
- 1) Understanding the contents — 작품/주제 이해도
- 2) Ideas & Argumentation — 주장·근거·통찰
- 3) Composition & Organization — 구조·논리 흐름
- 4) Expression & Sentences — 어휘·맞춤법·문장 다양성
- 5) Attitude & Integrity — 분량·성찰·지침 준수

**Feedback requirements:**
1) Corrections (spell, spacing, grammar): list every error as objects with fields `original`, `amended`, `reason`.  
2) Praise: concrete compliments referencing exact phrases/parts from the student's text (no vague praise).  
3) Improvements: clear, specific fixes with a concrete `rewrite_example` for each issue.  
4) Scores: integer 1–5 for each rubric item with short `explanation` per item, and `overall_score` (1–5).  
5) Overall comment: brief summary.

**Language for outputs:** {lang}

Student essay begins below (between triple backticks):
```essay
{essay}
```

Return **ONLY** a single JSON object with this schema:
{
  "scores": {
    "understanding": {"score": int, "explanation": string},
    "ideas_arguments": {"score": int, "explanation": string},
    "organization": {"score": int, "explanation": string},
    "expression": {"score": int, "explanation": string},
    "attitude_integrity": {"score": int, "explanation": string}
  },
  "overall_score": int,
  "corrections": [ {"original": string, "amended": string, "reason": string}, ... ],
  "praise": [ string, ... ],
  "improvements": [ {"issue": string, "suggestion": string, "rewrite_example": string}, ... ],
  "overall_comment": string
}
"""


def build_user_prompt(topic_label: str, custom_topic: str, essay: str, lang_choice: str) -> str:
    topic_final = custom_topic.strip() if topic_label.startswith("Custom") else topic_label
    lang_map = {
        "Korean": "Korean (한국어)",
        "English": "English",
        "Bilingual": "Bilingual (한국어+English)"
    }
    return USER_INSTRUCTIONS_TEMPLATE.format(
        topic=topic_final,
        essay=essay,
        lang=lang_map.get(lang_choice, "Korean"),
    )


# ----------------------------
# OpenAI Call
# ----------------------------

def call_openai(prompt: str, api_key: str, model: str) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    text = completion.choices[0].message.content
    return json.loads(text)


# ----------------------------
# UI: Evaluate Button
# ----------------------------

def render_rubric_table(result: Dict[str, Any]):
    st.subheader("📊 Rubric Scores (1–5)")
    rows = []
    for r in RUBRIC:
        item = result.get("scores", {}).get(r["id"], {})
        score = item.get("score", "-")
        expl = item.get("explanation", "")
        rows.append({"Item": r["name"], "Score": score, "Band": RUBRIC_SCALE.get(score, "-"), "Explanation": expl})
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_corrections(result: Dict[str, Any]):
    st.subheader("🛠️ Corrections (Original → Amended)")
    corrections: List[Dict[str, str]] = result.get("corrections", [])
    if not corrections:
        st.info("No corrections detected.")
        return
    for i, c in enumerate(corrections, start=1):
        with st.expander(f"Correction {i}"):
            st.markdown(f"**Original**: {c.get('original','')}")
            st.markdown(f"**Amended**: {c.get('amended','')}")
            st.markdown(f"**Reason**: {c.get('reason','')}")


def render_list_section(title: str, items: List[str], empty_msg: str):
    st.subheader(title)
    if not items:
        st.info(empty_msg)
        return
    for it in items:
        st.markdown(f"- {it}")


if st.button("🚀 Evaluate", type="primary", use_container_width=True):
    if not api_key:
        st.error("Please provide an OpenAI API key in the sidebar.")
    elif not essay.strip():
        st.warning("Please paste a student's essay.")
    else:
        with st.spinner("Evaluating with LLM…"):
            try:
                prompt = build_user_prompt(topic, custom_topic, essay, lang)
                result = call_openai(prompt, api_key, model_name)

                # Top summary
                st.success("Evaluation complete.")
                st.markdown(f"**Overall Score**: {result.get('overall_score', '-')}/5")
                st.markdown(result.get("overall_comment", ""))

                # Rubric table
                render_rubric_table(result)

                # Praise
                render_list_section("🌟 What You Did Well (Praise)", result.get("praise", []), "No praise items returned.")

                # Improvements
                improvements = result.get("improvements", [])
                st.subheader("🔧 Improvements (with concrete rewrites)")
                if not improvements:
                    st.info("No improvement suggestions returned.")
                else:
                    for i, imp in enumerate(improvements, start=1):
                        with st.expander(f"Improvement {i}"):
                            st.markdown(f"**Issue**: {imp.get('issue','')}")
                            st.markdown(f"**Suggestion**: {imp.get('suggestion','')}")
                            st.markdown(f"**Rewrite Example**:\n\n> {imp.get('rewrite_example','')}")

                # Corrections
                render_corrections(result)

                # Download raw JSON
                st.download_button(
                    label="⬇️ Download JSON Feedback",
                    data=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name="evaluation_feedback.json",
                    mime="application/json",
                )

            except Exception as e:
                st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption(
    "© 2025 Student Writing Evaluator • Built with Streamlit + OpenAI. This tool follows your rubric and generates exhaustive, example-based feedback."
)
