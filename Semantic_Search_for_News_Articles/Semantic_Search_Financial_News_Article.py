import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st

df = pd.read_csv("IndianFinancialNewsArticles(2003-2020).csv")
df = df.head(500)

model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L6-cos-v5')
embeddings = model.encode(df['Description'].tolist())

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

index.ntotal

faiss.write_index(index, "news_index")

def search(query):
    query_emb = model.encode([query])
    k = 10
    D, I = index.search(query_emb, k)
    
#     st.title("News Article Search")
    st.write("Search Results:")
    for i in range(len(I[0])):
        result_index = I[0][i]
        st.write(f"ID: {result_index}")
        st.write(f"Title: {df.loc[result_index]['Title']}")
        st.write(f"Date: {df.loc[result_index]['Date']}")
        st.write(f"Description: {df.loc[result_index]['Description']}")
        st.write("---")

if __name__ == "__main__":
    st.title("News Article Search")
    query = st.text_input("Enter your query:")
    if st.button("Search"):
        search(query)
