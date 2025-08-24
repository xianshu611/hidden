import os
import json
import streamlit as st
from openai import OpenAI

# -----------------------------
# Streamlit App Config
# -----------------------------
st.set_page_config(page_title="Hidden Figures 글감상 평가기", page_icon="🎬", layout="wide")
st.title("🎬 Hidden Figures 영화 감상문 자동 평가")
st.caption("Junior High Korean Writing Evaluator · GPT-4o mini · OpenAI SDK 1.x")

# Optional: 깃허브 리포 링크 영역
st.sidebar.header("설정")
st.sidebar.markdown(
    "[🔗 GitHub Repository (예시)](https://github.com/your-org/hidden-figures-evaluator)"
)
api_key = st.sidebar.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))

st.markdown("학생이 작성한 **〈히든 피겨스(Hidden Figures)〉 감상문**을 입력하고 **분석하기**를 누르세요.")

student_text = st.text_area(
    "감상문 입력",
    height=300,
    placeholder="예) 영화의 줄거리와 주인공들의 역할, 당시 사회적 배경 속 기술의 의미를 바탕으로 나의 생각을 정리해 보았습니다...",
)

# -----------------------------
# OpenAI Client
# -----------------------------
def get_client(key: str) -> OpenAI:
    if not key:
        st.warning("OPENAI_API_KEY가 필요합니다. 사이드바에서 입력해 주세요.")
        return None
    os.environ["OPENAI_API_KEY"] = key
    return OpenAI()

# -----------------------------
# 평가 프롬프트 (시스템 역할)
# -----------------------------
SYSTEM_PROMPT = """\
당신은 중학생 글쓰기 평가 전문가이자 한국어 교정 전문가입니다.
학생의 〈히든 피겨스(Hidden Figures)〉 영화 감상문을 아래 [채점 기준]에 따라 평가하고,
[피드백 지침]을 준수하여 결과를 한국어로 제공합니다.

[채점 기준]
1. 내용 이해(영화 줄거리·인물의 역할 이해, 주제 파악)
 - 4: 기술과 사회 변화를 넓게 이해하고 다양한 관점으로 서술
 - 3: 구체적 사례를 들어 중요성을 설명
 - 2: 중요성을 단순히 언급
 - 1: 의미가 모호하거나 오해
2. 사고력·창의성(독자적 관점, 설득력)
 - 4: 독창적 관점과 설득력 있는 전개
 - 3: 부분적으로 새로움·설득 요소 존재
 - 2: 일반적·단순한 의견
 - 1: 주장이나 아이디어가 거의 없음
3. 구성·조직(글 구조, 논리 흐름)
 - 4: 도입-전개-결론이 분명하고 논리적 연결
 - 3: 구조는 있으나 반복·연결 약함
 - 2: 단락 구성이 불분명하거나 흐름 이탈
 - 1: 흐름이 산만하고 단락 구성 미흡
4. 표현·문장(어휘, 맞춤법, 문장 다양성)
 - 4: 어휘·문장 풍부, 맞춤법 정확
 - 3: 전반 정확하나 일부 오류
 - 2: 반복적 표현·맞춤법 오류 다수
 - 1: 표현 부족으로 이해 어려움
5. 태도·성실성(분량, 주제 성실성)
 - 4: 분량 충분, 주제를 끝까지 성실히 탐구
 - 3: 적절한 분량, 전반적으로 충실
 - 2: 분량 부족 또는 부분 이탈
 - 1: 분량 부족, 주제에서 크게 이탈

[피드백 지침]
1) 포착(교정): “원문 → 수정문”을 제시하고, 수정 이유를 간단히 설명합니다.
   - 맞춤법·띄어쓰기·문법 오류를 포함합니다.
2) 잘한 점: 구체적 문장을 인용하여 칭찬합니다(막연한 표현 금지).
3) 보완점: 지적에 그치지 말고 “무엇을 어떻게 고칠지”를 구체적 대안으로 제시합니다.

[출력 형식]
JSON으로만 응답하세요. 키는 다음 네 개를 반드시 포함합니다.
{
  "grade": "A+|A0|B+|B0|C+|C0|D+|D0|F",
  "corrections": [
    {"original": "원문 일부", "revised": "수정문", "reason": "수정 이유"},
    ...
  ],
  "praise": [
    {"quote": "잘된 문장/내용 인용", "reason": "왜 좋은지"},
    ...
  ],
  "improvements": [
    {"issue": "보완이 필요한 점", "suggestion": "구체적 수정/추가 방안"},
    ...
  ]
}
반드시 위 JSON 스키마를 지키고, 추가 텍스트는 출력하지 마세요.
"""

# -----------------------------
# LLM 호출 함수
# -----------------------------
def evaluate_essay(client: OpenAI, text: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "다음은 학생의 〈히든 피겨스〉 감상문입니다.\n\n"
                f"=== 학생 글 시작 ===\n{text}\n=== 학생 글 끝 ===\n\n"
                "위 [채점 기준] 및 [피드백 지침]을 적용해 JSON만 출력하세요."
            ),
        },
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
    )
    content = resp.choices[0].message.content.strip()

    # JSON 파싱
    try:
        data = json.loads(content)
        return data
    except json.JSONDecodeError:
        # JSON 실패 시 간단 복구 시도
        content_fixed = content.strip("` \n")
        if content_fixed.startswith("{") and content_fixed.endswith("}"):
            try:
                return json.loads(content_fixed)
            except Exception:
                pass
        # 최종 실패: 원문 반환
        return {"raw": content}

# -----------------------------
# UI 동작
# -----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🔎 분석하기", use_container_width=True):
        if not student_text.strip():
            st.error("감상문을 입력해 주세요.")
        else:
            client = get_client(api_key)
            if client:
                with st.spinner("LLM이 평가 중입니다..."):
                    result = evaluate_essay(client, student_text)

                st.subheader("결과")

                if "raw" in result:
                    st.warning("JSON 파싱에 실패하여 원문을 표시합니다.")
                    st.code(result["raw"], language="json")
                else:
                    # 1. 평가 등급
                    grade = result.get("grade", "N/A")
                    st.markdown(f"**1) 평가 등급:** `{grade}`")

                    # 2. 포착(교정)
                    st.markdown("**2) 포착(교정)**")
                    corrections = result.get("corrections", [])
                    if corrections:
                        for i, c in enumerate(corrections, 1):
                            st.markdown(
                                f"- {i}. **원문**: {c.get('original','')}\n"
                                f"   \n   **수정문**: {c.get('revised','')}\n"
                                f"   \n   **이유**: {c.get('reason','')}"
                            )
                    else:
                        st.write("표시할 교정 항목이 없습니다.")

                    # 3. 잘한 점
                    st.markdown("**3) 잘한 점**")
                    praise = result.get("praise", [])
                    if praise:
                        for i, p in enumerate(praise, 1):
                            st.markdown(
                                f"- {i}. **인용**: {p.get('quote','')}\n"
                                f"   \n   **이유**: {p.get('reason','')}"
                            )
                    else:
                        st.write("표시할 칭찬 항목이 없습니다.")

                    # 4. 보완점
                    st.markdown("**4) 보완점 (구체적 대안 포함)**")
                    improvements = result.get("improvements", [])
                    if improvements:
                        for i, im in enumerate(improvements, 1):
                            st.markdown(
                                f"- {i}. **이슈**: {im.get('issue','')}\n"
                                f"   \n   **제안**: {im.get('suggestion','')}"
                            )
                    else:
                        st.write("표시할 보완 항목이 없습니다.")

with col2:
    st.markdown("### 📌 사용 안내")
    st.markdown(
        """
- 입력: 〈히든 피겨스〉 감상문 전체 내용을 붙여넣으세요.
- 출력: **등급**, **포착(교정)**, **잘한 점**, **보완점**을 구조적으로 제공합니다.
- 모델: `gpt-4o-mini` (OpenAI SDK 1.x)
- 안전: 민감한 개인정보는 입력하지 마세요.
        """
    )
    st.markdown("---")
    st.markdown("### ⚙️ 개발 메모")
    st.code(
        'resp = client.chat.completions.create(...)\n'
        'content = resp.choices[0].message.content',
        language="python",
    )
