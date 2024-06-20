import os
from langchain_community.document_loaders import PyPDFLoader

def read_pdfs(file_path:str)->list:
    
    
    directory_path = file_path
    all_files = os.listdir(directory_path)
    pdf_files = [file for file in all_files if file.endswith('.pdf')]
    all_pages = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory_path, pdf_file)
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        all_pages.extend(pages)

    return all_pages
