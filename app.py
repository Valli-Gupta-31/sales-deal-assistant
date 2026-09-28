import os
import re

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from hindsight_client import Hindsight

load_dotenv()

GROQ_MODELS = (
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "qwen/qwen3.8-27b",
)
RECALL_QUERY = "What were their main objections and concerns?"
DEFAULT_HINDSIGHT_URL = "https://api.hindsight.vectorize.io"


def _env(name: str, default: str | None = None) -> str:
    return (os.getenv(name) or default or "").strip()


def bank_id_from_name(prospect_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", prospect_name.strip().lower()).strip("-")
    return slug or "prospect"


def format_recall_context(recall_response) -> str:
    results = getattr(recall_response, "results", None) or []
    lines = []
    for item in results:
        text = getattr(item, "text", None) or str(item)
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines)


@st.cache_resource
def get_clients() -> tuple[Hindsight, Groq]:
    groq_key = _env("GROQ_API_KEY")
    hindsight_key = _env("HINDSIGHT_API_KEY")
    hindsight_url = _env("HINDSIGHT_API_URL", DEFAULT_HINDSIGHT_URL)

    missing = []
    if not groq_key:
        missing.append("GROQ_API_KEY")
    if not hindsight_key:
        missing.append("HINDSIGHT_API_KEY")
    if missing:
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}. "
            "Add them to your .env file."
        )

    hindsight = Hindsight(base_url=hindsight_url, api_key=hindsight_key)
    groq_client = Groq(api_key=groq_key)
    return hindsight, groq_client


def generate_follow_up_email(
    groq_client: Groq,
    model: str,
    prospect_name: str,
    recalled_context: str,
) -> str:
    system_prompt = (
        "You are a skilled B2B sales assistant. Write a concise, personalized "
        "follow-up email that directly addresses the prospect's past objections "
        "and concerns. Be professional, specific, and helpful—never pushy. "
        "Use only the recalled meeting notes as facts. If a detail is missing, "
        "do not invent it. Return only the email body with a subject line."
    )
    user_prompt = (
        f"Prospect name: {prospect_name}\n\n"
        f"Recalled objections and concerns:\n{recalled_context}\n\n"
        "Write a personalized follow-up / pitch email that addresses these points."
    )
    completion = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )
    message = completion.choices[0].message
    content = (message.content or "").strip()
    if content:
        return content
    reasoning = getattr(message, "reasoning", None)
    if reasoning:
        return str(reasoning).strip()
    return ""


st.set_page_config(
    page_title="Sales Deal Intelligence",
    page_icon="◆",
    layout="centered",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

      html, body, [class*="css"] {
        font-family: "IBM Plex Sans", sans-serif;
      }

      .stApp {
        background: linear-gradient(180deg, #f4f1eb 0%, #ece8e0 100%);
      }

      [data-testid="stHeader"] {
        background: transparent;
      }

      .hero {
        padding: 0.4rem 0 1.4rem 0;
      }
      .hero h1 {
        font-family: "DM Sans", sans-serif;
        font-size: 2.05rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: #1c1917;
        margin: 0 0 0.35rem 0;
      }
      .hero p {
        color: #57534e;
        margin: 0;
        font-size: 1.02rem;
      }

      .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
        padding: 0;
        border: 0;
      }
      .stTabs [data-baseweb="tab"] {
        color: #57534e;
        font-weight: 600;
      }

      .stTextInput input, .stTextArea textarea, [data-baseweb="select"] {
        border-radius: 10px !important;
      }

      .stButton > button {
        width: 100%;
        border-radius: 12px;
        background: #1c1917;
        color: #fafaf9;
        border: 0;
        font-weight: 600;
        padding: 0.55rem 1rem;
      }
      .stButton > button:hover {
        background: #292524;
        color: #fff;
      }

      .memory-card, .email-card {
        background: #fff;
        border: 1px solid #e7e2d9;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-top: 0.75rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Sales Deal Intelligence</h1>
      <p>Log meeting notes, recall objections, and generate a tailored follow-up with Groq + Hindsight.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    hindsight, groq_client = get_clients()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.subheader("Model")
    selected_model = st.selectbox(
        "Groq model",
        GROQ_MODELS,
        index=0,
        help="Used when generating the follow-up email.",
    )
    st.caption(
        "Keys are loaded from `.env` (`GROQ_API_KEY`, `HINDSIGHT_API_KEY`, optional `HINDSIGHT_API_URL`)."
    )

log_tab, pitch_tab = st.tabs(["Log Meeting Notes / Objections", "Generate Follow-Up / Pitch"])

with log_tab:
    st.markdown("Store what you heard on the call. Each prospect gets their own Hindsight memory bank.")
    with st.form("log_meeting_form", clear_on_submit=False):
        prospect_name = st.text_input(
            "Prospect name",
            key="log_prospect",
            placeholder="e.g. Priya Shah",
        )
        company = st.text_input("Company", placeholder="e.g. Northwind Analytics")
        objections = st.text_area(
            "Objections raised",
            placeholder="Price, implementation time, need legal review…",
            height=110,
        )
        features = st.text_area(
            "Key features wanted",
            placeholder="SSO, audit logs, Salesforce sync…",
            height=110,
        )
        extra_notes = st.text_area(
            "Other notes (optional)",
            placeholder="Timeline, stakeholders, next steps…",
            height=90,
        )
        save_notes = st.form_submit_button("Save to Hindsight", type="primary")

    if save_notes:
        if not prospect_name.strip():
            st.warning("Enter a prospect name so notes can be stored in their bank.")
        elif not any(part.strip() for part in (objections, features, extra_notes, company)):
            st.warning("Add at least one meeting detail before saving.")
        else:
            input_text = (
                f"Prospect: {prospect_name.strip()}\n"
                f"Company: {company.strip() or 'Unknown'}\n"
                f"Objections raised: {objections.strip() or 'None recorded'}\n"
                f"Key features wanted: {features.strip() or 'None recorded'}\n"
                f"Other notes: {extra_notes.strip() or 'None'}"
            )
            bank_id = bank_id_from_name(prospect_name)
            with st.spinner("Retaining meeting notes…"):
                try:
                    hindsight.retain(bank_id=bank_id, content=input_text)
                except Exception as exc:
                    st.error(f"Could not retain notes: {exc}")
                else:
                    st.success(f"Saved notes for **{prospect_name.strip()}** (bank `{bank_id}`).")
                    st.code(input_text, language="text")

with pitch_tab:
    st.markdown("Recall stored objections for a prospect, then draft a personalized email with Groq.")
    with st.form("pitch_form", clear_on_submit=False):
        pitch_prospect = st.text_input(
            "Prospect name",
            key="pitch_prospect",
            placeholder="e.g. Priya Shah",
        )
        generate_email = st.form_submit_button("Generate follow-up email", type="primary")

    if generate_email:
        if not pitch_prospect.strip():
            st.warning("Enter the prospect name used when logging notes.")
        else:
            bank_id = bank_id_from_name(pitch_prospect)
            recall_response = None
            with st.spinner("Recalling objections from Hindsight…"):
                try:
                    recall_response = hindsight.recall(
                        bank_id=bank_id,
                        query=RECALL_QUERY,
                    )
                except Exception as exc:
                    st.error(f"Could not recall memories: {exc}")

            if recall_response is not None:
                recalled_context = format_recall_context(recall_response)
                if not recalled_context:
                    st.info(
                        f"No memories found for **{pitch_prospect.strip()}** "
                        f"(bank `{bank_id}`). Log meeting notes first, then try again in a few seconds."
                    )
                else:
                    st.markdown("**Recalled objections & concerns**")
                    st.markdown(
                        f'<div class="memory-card">{recalled_context.replace(chr(10), "<br>")}</div>',
                        unsafe_allow_html=True,
                    )

                    with st.spinner(f"Drafting email with `{selected_model}`…"):
                        try:
                            email = generate_follow_up_email(
                                groq_client,
                                selected_model,
                                pitch_prospect.strip(),
                                recalled_context,
                            )
                        except Exception as exc:
                            st.error(f"Groq could not generate the email: {exc}")
                        else:
                            if not email:
                                st.error("The model returned an empty response. Try the other Groq model.")
                            else:
                                st.markdown("**Personalized follow-up**")
                                st.text_area(
                                    "Email draft",
                                    value=email,
                                    height=280,
                                    label_visibility="collapsed",
                                )
