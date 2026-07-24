import os
import pickle
import subprocess
from req_res import Request, Response
#from google import genai
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
#import google.generativeai as genai



def init_llm_model(api_key=None):
    if api_key is None:
        raise ValueError("API Key is required")

    # Initializes Groq with a fast, high-quality llama model
    model = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key)
    return model




# def init_llm_model(api_key=None):
#     if api_key is None:
#         raise ValueError("API Key is required")

#     genai.configure(api_key=api_key)
#     model = genai.GenerativeModel("models/gemini-2.5-flash")
#     return model


def embedding_model():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",#"all-mpnet-base-v2"
        model_kwargs={"device": "cpu"}
    )
  


# def load_index(filename, force_rebuild_index=False):
#     if force_rebuild_index or not os.path.exists(filename):
#         print("Force Rebuilding Index...")
#         cmd = "python build_index.py"
#         subprocess.call(cmd, shell=True)

#     with open(filename, "rb") as f:
#         vector_store = pickle.load(f)

#     return vector_store




def load_index(filename, force_rebuild_index=False):
    # Change filename check to look for the native directory path
    dir_name = "./data/faiss_index"
    if force_rebuild_index or not os.path.exists(dir_name):
        print("Force Rebuilding Index...")
        cmd = "python build_index.py"
        subprocess.call(cmd, shell=True)

    # Use native LangChain FAISS loader instead of pickle
    vector_store = FAISS.load_local(dir_name, embedding_model(), allow_dangerous_deserialization=True)
    return vector_store





#Hypothetical Document Embedding (HyDE) in Document Retrieval


class HyDERetriever:
    def __init__(self, API_KEY2 , chunk_size=500, chunk_overlap=100 ):
        # Change this line to use ChatGroq:
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=API_KEY2)

        self.embeddings = embedding_model()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # with open("data/vector_store.pkl", "rb") as f:
        #     self.vector_store = pickle.load(f)    
        
        
        # Native FAISS loader for the retriever block
        self.vector_store = FAISS.load_local("./data/faiss_index", self.embeddings, allow_dangerous_deserialization=True)
        
        self.hyde_prompt = PromptTemplate(
            input_variables=["query", "chunk_size"],
            template = """
You are generating a hypothetical document for retrieval (HyDE step).

Given the question:
{query}

Generate a detailed, clear, and structured explanation that directly answers the question based ONLY on general knowledge of the Ayushman registration process.

Rules:
- Be factual and deterministic (no creativity, no storytelling style).
- Focus on key concepts, roles, and relationships.
- Do NOT add imaginary or poetic content.
- Keep the tone informational (like a textbook explanation).
- Ensure the content is relevant for semantic search.

The document should be approximately {chunk_size} characters long.
"""
        )
        self.hyde_chain = self.hyde_prompt | self.llm

    def generate_hypothetical_document(self, query):
        input_variables = {"query": query, "chunk_size": self.chunk_size}
        return self.hyde_chain.invoke(input_variables).content

    def retrieve(self, query, k=3):
        hypothetical_doc = self.generate_hypothetical_document(query)
        similar_docs = self.vector_store.similarity_search_with_relevance_scores(hypothetical_doc, k=5)
        return similar_docs, hypothetical_doc