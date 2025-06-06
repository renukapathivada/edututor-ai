import streamlit as st
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# ------------------------------
# Firebase Setup
# ------------------------------
cred_dict = dict(st.secrets["firebase"])
cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")  # Fix escaped newlines
cred = credentials.Certificate(cred_dict)

# Initialize Firebase only once
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://edututor-ai-370b5-default-rtdb.asia-southeast1.firebasedatabase.app/"
    })

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="EduTutor AI", layout="wide")

# ------------------------------
# Load Models Once
# ------------------------------
@st.cache_resource
def load_models():
    gen_model = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(gen_model)
    lm = AutoModelForSeq2SeqLM.from_pretrained(gen_model)
    generator = pipeline("text2text-generation", model=lm, tokenizer=tokenizer)
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return generator, semantic_model

generator, semantic_model = load_models()

# ------------------------------
# Sidebar Role Selection
# ------------------------------
role = st.sidebar.selectbox("Login As", ["Student", "Teacher"])

# ------------------------------
# Student View
# ------------------------------
if role == "Student":
    st.header("📘 Student Portal")
    name = st.text_input("Name")
    topic = st.text_input("Topic (e.g., Photosynthesis)")
    style = st.selectbox("Learning Style", ["Visual", "Auditory", "Hands‑on"])

    if st.button("Generate Lesson & Quiz") and name and topic:
        with st.spinner("Generating content..."):
            lesson_prompt = f"Explain {topic} simply for a {style} learner."
            lesson = generator(lesson_prompt, max_length=200)[0]["generated_text"]

            quiz_prompt = f"Create a short-answer question about {topic}."
            question = generator(quiz_prompt, max_length=100)[0]["generated_text"]

        st.subheader("📘 Lesson")
        st.write(lesson)

        st.subheader("📝 Quiz Question")
        st.write(question)

        answer = st.text_area("Your Answer")
        if st.button("Submit Answer") and answer:
            with st.spinner("Evaluating answer..."):
                ideal_prompt = f"Perfect short answer to: {question}"
                ideal_ans = generator(ideal_prompt, max_length=100)[0]["generated_text"]

                emb = semantic_model.encode([ideal_ans, answer], convert_to_tensor=True)
                score = util.pytorch_cos_sim(emb[0], emb[1]).item()

                if score > 0.85:
                    outcome = "✅ Excellent!"
                elif score > 0.65:
                    outcome = "👍 Good."
                elif score > 0.4:
                    outcome = "🧐 Needs more detail."
                else:
                    outcome = "⚠️ Review and try again."

            st.markdown(f"**Similarity Score:** {score:.2f}")
            st.success(f"Feedback: {outcome}")

            db.reference("students").push({
                "name": name,
                "topic": topic,
                "question": question,
                "answer": answer,
                "score": round(score * 100, 2),
                "feedback": outcome,
                "timestamp": datetime.now().isoformat()
            })

# ------------------------------
# Teacher View
# ------------------------------
elif role == "Teacher":
    st.header("🧑‍🏫 Teacher Dashboard")
    data = db.reference("students").get()

    if data:
        records = list(data.values())
        df = pd.DataFrame(records)
        st.subheader("📊 Student Submissions")
        st.dataframe(df)

        if "score" in df.columns and "topic" in df.columns:
            st.subheader("📈 Average Score by Topic")
            chart_data = df.groupby("topic")["score"].mean().reset_index()
            st.bar_chart(chart_data.set_index("topic"))
        else:
            st.warning("No score data available for chart.")
    else:
        st.info("No student submissions yet.")
