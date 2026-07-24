import os
import sys
import gc

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.parsers.images import RapidOCRBlobParser

# --- EMBEDDING SETUP ---
from langchain_community.embeddings import HuggingFaceEmbeddings
def embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

dataset_path = '/kaggle/input/datasets/pavankurman/pm-jay-registrations/PM_JAY_REGISTRATION'
output_dir = "./data"
os.makedirs(output_dir, exist_ok=True)

num_files = 21
index_top_k = True  
cache_index = True

print(f"Indexing top k files: {num_files}, index_top_k: {index_top_k}")

# --- DUAL CHANNELS SPLITTING FUNCTION ---
def split_file_to_chunks(file_path, filename):
    # Skip OCR scanning for the 240+ page file to prevent Kaggle RAM crash limits
    if "operational_guidelines" in filename.lower() or os.path.getsize(file_path) > 12 * 1024 * 1024:
        print(f" Giant Document Found ({os.path.getsize(file_path)/(1024*1024):.1f}MB). Reading native text layers only.")
        loader = PyMuPDFLoader(file_path, extract_images=False)
    else:
        # Secure OCR pipeline active for forms, user guides, and screenshots
        loader = PyMuPDFLoader(
            file_path, 
            extract_images=True, 
            images_parser=RapidOCRBlobParser()
        )
    
    try:
        document = loader.load()
    except Exception as ocr_err:
        print(f"OCR pipeline bypass notice: {ocr_err}")
        loader = PyMuPDFLoader(file_path, extract_images=False)
        document = loader.load()
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents=document)
    
    del loader
    del document
    return chunks

all_chunks = []
count = 0
early_exit = False

print("\n--- Starting Hybrid Text + Visual OCR Processing ---")
for folder_name, _, filename in os.walk(dataset_path):
    for file in filename:
        if file.lower().endswith(".pdf") and not file.startswith('.'):
            file_path = os.path.join(folder_name, file)
            print("Loading: ", file)
            
            chunks = split_file_to_chunks(file_path, file)
            all_chunks.extend(chunks)
            count += 1
            
            gc.collect() 
            
            if index_top_k and count >= num_files:
                early_exit = True
                break
    if early_exit:
        break

print("\nTotal chunks extracted:", len(all_chunks))
if not all_chunks:
    raise ValueError("No documents processed! Check your Kaggle dataset path.")
    
print("\nSample chunk preview:\n", all_chunks[0].page_content[:300])
print("\nBuilding FAISS Vector Index...")

vector_store = FAISS.from_documents(all_chunks, embedding=embedding_model())

if cache_index:
    vector_store.save_local(os.path.join(output_dir, "faiss_index"))
    print(f"Index successfully exported to: {output_dir}/faiss_index")
    
print("Done... Complete pipeline finished successfully!")