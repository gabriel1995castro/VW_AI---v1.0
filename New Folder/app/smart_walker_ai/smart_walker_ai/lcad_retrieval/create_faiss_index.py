#A script to create a faiss index with multiple books and pdfs

import json
from pypdf import PdfReader
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time
import os

index_name = 'chunks.index'
index_dir = 'data'
df_corpus_path_dir = 'data'
df_corpus_name = 'df_corpus.csv'
texts_path_dir = 'texts'

text = ""
paragraph_list = []
paragraph_size = 1000 #4000 for df_corpus_long
paragraph_overlap = 80 #60 for df_corpus_long_test

#Read pdf to .txt
def pdf_to_text(texts_path_dir: bool = texts_path_dir, verbose = False):
    def substitute_symbols(original_str):
        return original_str.replace("/C0", "-").replace("/C3", "*").replace("/C14", "°")
    if verbose:
        print("Converting pdfs in ", texts_path_dir, " to .txt...")
    for filename in os.listdir(texts_path_dir):
        try:
            text = ''
            if filename.endswith(".pdf"):
                reader = PdfReader(os.path.join(texts_path_dir, filename))
                for page in reader.pages:
                    #print(f"\n\n ------PAGE {i}------ \n\n")
                    text += substitute_symbols(page.extract_text()) + "\n"
                #print(text[4000:6000])
                with open(os.path.join(texts_path_dir, filename.replace('.pdf', '.txt')), 'w') as f:
                    f.write(text) 
        except Exception as e:
            print("Error: ", str(e))

def create_df_corpus_from_txt(texts_path_dir: str = texts_path_dir, 
                     paragraph_size: int = paragraph_size, 
                     paragraph_overlap: int = paragraph_overlap, 
                     sample_size: int = None,
                     verbose: bool = False) -> pd.DataFrame:
    def num_char(chunk):
        """ Calculate the total number of characters in the given chunk."""
        return sum(len(line[1]) for line in chunk)

    def create_dataframe_from_text(text, filename='NA', chunk_size=paragraph_size, overlap_percent=paragraph_overlap):
        """ Creates a pandas DataFrame from a given text file."""
        #create list with tuple (line, text)
        lines = text.split('\n')
        lines = [(i, line) for i, line in enumerate(lines)]

        #create list with groups tuple (initial_line, text)
        chunks = []
        chunk = []
        for (line, text) in lines:
            chunk.append((line, text))
            if num_char(chunk) > chunk_size:
                chunks.append((chunk[0][0], '\n'.join(t[1] for t in chunk), filename))
                chunk = chunk[int(len(chunk)*(1-(overlap_percent/100))):]
        if len(chunk) > 0:
            chunks.append((chunk[0][0], '\n'.join(t[1] for t in chunk), filename))

        return pd.DataFrame(chunks, columns =['init_line', 'text', 'source'])
    if verbose:
        print("Creating dataframe from .txt files in ", texts_path_dir, "...")

    df_corpus = pd.DataFrame(columns=['init_line', 'text', 'source'])
    for filename in os.listdir(texts_path_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(texts_path_dir, filename), 'r') as f:
                text = f.read()
                df_corpus = pd.concat([df_corpus, \
                                       create_dataframe_from_text(
                                           text, 
                                           filename, 
                                           chunk_size=paragraph_size, 
                                           overlap_percent=paragraph_overlap
                                        )], ignore_index=True)
    if sample_size is not None:
        df_corpus = df_corpus.sample(sample_size)
    df_corpus.reset_index(inplace=True)
    # df_corpus.to_csv(df_corpus_path_dir+'/'+df_corpus_name, index=False, escapechar="\\")
    return df_corpus


def create_df_corpus_from_json(
    json_file_path: str,  
    json_key_path: list[str] = ["area_layout", "offices"],
    json_indexing_keys: list[str] = ["name"],
) -> pd.DataFrame:
    with open(json_file_path) as f:
        input_json = json.load(f)
    input_list = input_json
    for key in json_key_path:
        input_list = input_list[key]
    chunks = []
    for idx, entry in enumerate(input_list):
        indexing_str = ""
        for key in json_indexing_keys:
            indexing_str += f"{key}: {entry[key]}\n"
        chunks.append({
            "init_line": idx,
            "reference": indexing_str,
            "text": str(entry),
            "source": json_file_path
        })
    return pd.DataFrame(chunks, columns =['init_line', 'reference', 'text', 'source'])
        
            


def create_index(
        df_corpus_path_dir: str = df_corpus_path_dir, 
        df_corpus_name: str = df_corpus_name,
        encoder_model: SentenceTransformer = None,  
        index_column: str = "text",
        index_dir: str = index_dir,
        index_name: str = index_name,
        encoder_model_dimensions: int = 1024,
        verbose: bool = False
    ):
    if encoder_model is None:
        raise ValueError("Encoder model must be provided.")
    df_corpus: pd.DataFrame = pd.read_csv(df_corpus_path_dir+'/'+df_corpus_name)
    if verbose:
        print("Creating index from ", df_corpus_path_dir+'/'+df_corpus_name)
    #Creation of index
    t=time.time()
    text_list = [str(text) for text in df_corpus[index_column].tolist()]
    encoded_data = encoder_model.encode(text_list, show_progress_bar=verbose)
    encoded_data = np.asarray(encoded_data.astype('float32'))
    index_blast_furnace = faiss.IndexIDMap(faiss.IndexFlatIP(encoder_model_dimensions)) #768
    index_blast_furnace.add_with_ids(encoded_data, np.array(range(0, len(df_corpus))))
    faiss.write_index(index_blast_furnace, index_dir+'/'+index_name)

    if verbose:
        print('>>>> Created embeddings in Total Time: {}'.format(time.time()-t)) 
        print('Index saved to: ', index_dir+'/'+index_name)
