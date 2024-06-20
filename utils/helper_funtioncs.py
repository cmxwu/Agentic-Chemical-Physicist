import os
from langchain_community.document_loaders import PyPDFLoader
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re

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


def get_impt_words(all_pages:list, topk: int)-> list[str]:

    all_pages_str = [page.page_content for page in all_pages]

    def remove_numbers(text):
        return re.sub(r'\d+', '', text)
    
    
    all_documents = [remove_numbers(doc) for doc in all_pages_str] 
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)  
    
    tfidf_matrix = vectorizer.fit_transform(all_documents)

    feature_names = vectorizer.get_feature_names_out()

    tfidf_dense = tfidf_matrix.todense()
    tfidf_df = pd.DataFrame(tfidf_dense, columns=feature_names)
    word_scores = tfidf_df.sum(axis=0)
    top_words = word_scores.nlargest(topk).index.tolist()
    
    return top_words
