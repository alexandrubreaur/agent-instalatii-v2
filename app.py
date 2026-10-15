import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone

st.set_page_config(page_title="Agent Instalatii PRO", layout="wide")
st.title("Asistent Normative Proiectare Instalatii")

google_key = st.secrets.get("GOOGLE_API_KEY", "")
pinecone_key = st.secrets.get("PINECONE_API_KEY", "")

if not google_key or not pinecone_key:
    st.warning("Adauga ambele chei API in setarile Streamlit pentru a continua.")
    st.stop()

genai.configure(api_key=google_key)
model = genai.GenerativeModel('gemini-1.5-flash')

pc = Pinecone(api_key=pinecone_key)
index = pc.Index("normative-instalatii")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Adreseaza o intrebare din normative..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            emb_intrebare = genai.embed_content(
                model="models/gemini-embedding-001",
                content=prompt
            )['embedding']

            rezultate = index.query(
                vector=emb_intrebare,
                top_k=5,
                include_metadata=True
            )

            context = ""
            for match in rezultate['matches']:
                meta = match['metadata']
                context += f"\n[Sursa: {meta['sursa']}, Pagina {meta['pagina']}]\n{meta['text']}\n"

            full_prompt = (
                "Esti un inginer expert in proiectarea instalatiilor pentru constructii. "
                "Raspunde la intrebare bazandu-te strict pe fragmentele furnizate mai jos. "
                "Citeaza intotdeauna sursa si pagina la finalul raspunsului.\n\n"
                f"FRAGMENTE EXTRASE:\n{context}\n\n"
                f"INTREBARE: {prompt}"
            )

            response = model.generate_content(full_prompt)
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"Eroare la procesare: {e}")
