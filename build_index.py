from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from models import load_index, init_llm_model,embedding_model
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import sys
import pickle


##############

output_file = "./data/vector_store.pkl"
#creating directory in one line if exists ok else creates on its own
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# todo: add argument parser
num_files = None
index_top_k = False
if len(sys.argv) > 1:     #checking whether any pdf's or there or not
    index_top_k = True   
    num_files = 10        # how many PDFs to index (if we run python script.py 10)

print(f"Indexing top k files: {num_files}, index_top_k: {index_top_k}")

cache_index = True
#creating an object of class HuggingFaceEmbeddings which is embeds the chunks
# embedding_model = HuggingFaceEmbeddings(
#     model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

#function which is used to split sentences/files into chunks using RecursiveCharacterTextSplitter
def split_file_to_chunks(file_path):
    loader = PyPDFLoader(file_path)#creating the object of class PyPDFLoader through which python can read
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50)#object of class RecursiveCharacterTextSplitter with size 500,overlap with 50
    chunks = text_splitter.split_documents(documents=document)#here splitting is taking place
    return chunks


all_chunks = []
count = 0
early_exit = False
for folder_name, _, filename in os.walk("PM_JAY_REGISTRATION"):#here we are providing the folder which iteration shuld happen
    for file in filename:
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_name, file)
            print("Loading: ", file_path)
            chunks = split_file_to_chunks(file_path)
            all_chunks.extend(chunks)
            count += 1
            #if it crosses num_files [lets say 5 pdf's in different folders and
            #if you give (python script.py 2) it will check 2 only it will not consider 5]
            if index_top_k and count >= num_files:
                early_exit = True
                break

    if early_exit:
        break
print("Total chunks:", len(all_chunks))
print("Sample chunk:\n", all_chunks[0].page_content[:300])
print("Indexing files")

if not all_chunks:
    raise ValueError("No documents found! Check your PDF folder.")
vector_store = FAISS.from_documents(
    all_chunks, embedding=embedding_model())#creating an object of FAISS for the database

if cache_index:
    print("Caching index")
    pickle.dump(vector_store, open(output_file, "wb"))

print("Done...")