import streamlit as st
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# Firebase setup
import json
cred = credentials.Certificate(json.loads(st.secrets["firebase"]))
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://edututor-ai-370b5-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

st.set_page_config(page_title="EduTutor AI", layout="wide")

# Load models once
@st.cache_resource
def load_models():
    gen_model = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(gen_model)
    lm = AutoModelForSeq2SeqLM.from_pretrained(gen_model)
    generator = pipeline("text2text-generation", model=lm, tokenizer=tokenizer)
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return generator, semantic_model

generator, semantic_model = load_models()

role = st.sidebar.selectbox("Login As", ["Student", "Teacher"])

if role == "Student":
    st.header("📘 Student Portal")
    name = st.text_input("Name")
    topic = st.text_input("Topic (e.g., Photosynthesis)")
    style = st.selectbox("Learning Style", ["Visual", "Auditory", "Hands‑on"])
    if st.button("Generate Lesson & Quiz"):
        lesson_prompt = f"Explain {topic} simply for a {style} learner."
        lesson = generator(lesson_prompt, max_length=200)[0]["generated_text"]
        st.subheader("📘 Lesson")
        st.write(lesson)

        quiz_prompt = f"Create a short-answer question about {topic}."
        question = generator(quiz_prompt, max_length=100)[0]["generated_text"]
        st.subheader("📝 Quiz Question")
        st.write(question)

        answer = st.text_area("Your Answer")
        if st.button("Submit Answer"):
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

elif role == "Teacher":
    st.header("🧑‍🏫 Teacher Dashboard")
    data = db.reference("students").get()
    if data:
        df = st.dataframe(list(data.values()))
        st.subheader("Performance Chart")
        chart_df = pd.DataFrame(list(data.values()))
        if "score" in chart_df.columns:
            st.bar_chart(chart_df.groupby("topic")["score"].mean())
    else:
        st.info("No student submissions yet.")
