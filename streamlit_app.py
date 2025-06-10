import streamlit as st
import pandas as pd
import numpy as np
import faiss
import json
import os
import openai
import spacy
from sentence_transformers import SentenceTransformer
st.set_page_config(page_title="LLM Fact Checker", layout="centered")
st.title("🔍 LLM Fact Checker")
st.markdown("Verify claims using trusted sources and GPT-4.")

# Load NLP & Embedding Models
def load_models():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    embed_model = SentenceTransformer('all-MiniLM-L6-v2') 
    return nlp, embed_model
nlp, embed_model = load_models()

# Load verified PIB facts
@st.cache_data
def load_facts():
    df = pd.read_csv("verified_facts.csv")
    return df['title'].tolist()

facts = load_facts()

# Build FAISS index
@st.cache_resource
def build_index(facts):
    embeddings = embed_model.encode(facts).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

index, embeddings = build_index(facts)

def extract_main_claim(text):
    doc = nlp(text)
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    named_entities = [ent.text for ent in doc.ents]
    key_claim = max(noun_chunks + named_entities, key=len) if noun_chunks + named_entities else text
    return key_claim.strip()

def retrieve_similar_facts(claim_text, top_k=3, min_score=0.7):
    claim_embedding = embed_model.encode([claim_text]).astype('float32')
    distances, indices = index.search(claim_embedding, top_k)

    valid_facts = []
    for dist, idx in zip(distances[0], indices[0]):
        score = 1 - dist / 2
        if score >= min_score:
            valid_facts.append(facts[idx])

    return valid_facts

def classify_claim_with_gpt(claim, retrieved_facts):
    if not retrieved_facts:
        return {
            "verdict": "Unverifiable",
            "evidence": [],
            "reasoning": "No semantically similar facts were retrieved to verify the claim."
        }

    prompt = f"""
You are an AI fact-checking assistant.

Claim: "{claim}"

Retrieved Facts:
{chr(10).join(f"- {fact}" for fact in retrieved_facts)}

Classify the claim as one of: "Likely True", "Likely False", or "Unverifiable".
Provide JSON output with keys: "verdict", "evidence", and "reasoning".
Respond only with JSON.
"""

    try:
        openai.api_key = os.getenv("API_KEY")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        output = response['choices'][0]['message']['content']
        return json.loads(output)

    except Exception as e:
        return {
            "verdict": "Error",
            "evidence": [],
            "reasoning": f"GPT classification failed: {str(e)}"
        }

user_input = st.text_area("Enter a short news claim or statement:")

if st.button("🔎 Verify Claim"):
    if user_input:
        with st.spinner("Extracting claim..."):
            claim = extract_main_claim(user_input)

        with st.spinner("Retrieving supporting facts..."):
            retrieved = retrieve_similar_facts(claim)

        with st.spinner("Running GPT-4 Fact Check..."):
            result = classify_claim_with_gpt(claim, retrieved)

        st.success(f"Verdict: **{result['verdict']}**")
        st.markdown("### Reasoning")
        st.write(result["reasoning"])
        st.markdown("### Evidence Used")
        for fact in result["evidence"]:
            st.markdown(f"- {fact}")

        st.download_button("Download JSON Result", data=json.dumps(result, indent=2), file_name="fact_check_result.json")
    else:
        st.warning("Please enter a claim to verify.")
