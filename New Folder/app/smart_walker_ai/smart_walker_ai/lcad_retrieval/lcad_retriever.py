#Semantic search using sbert and FAISS

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from faiss import read_index
import time

try:
    from smart_walker_ai.lcad_retrieval.create_faiss_index import create_df_corpus_from_json, create_df_corpus_from_txt, pdf_to_text, create_index
    from smart_walker_ai.lcad_retrieval.ollama_embedding import OllamaEmbedding
except Exception:
    from .create_faiss_index import create_df_corpus_from_json, create_df_corpus_from_txt, pdf_to_text, create_index
    from .ollama_embedding import OllamaEmbedding

CHUNKS_INDEX_NAME = 'chunks.index'
CHUNKS_DF_NAME = 'chunks.csv'

def unique(array):
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]

class SentenceRetriever:
    def __init__(
        self, 
        use_ollama: bool = False,
        encoder_model_id: str = "intfloat/multilingual-e5-large-instruct", 
        hf_token: str = "", 
    ):
        self.model = ''
        self.df_corpus = ''
        self.paragraphs_per_search = 1
        print("Loading encoder model: " + encoder_model_id)
        if use_ollama:
            self.model = OllamaEmbedding(encoder_model_id)
        else:
            self.model = SentenceTransformer(encoder_model_id, trust_remote_code=True, token=hf_token)

    def init_pdf_retriever(
        self, 
        input_dir: str, 
        data_dir: str = "retriever_data", 
        chunk_size: str = 1000, 
        overlap_chunks: str = 0, 
        force_rebuild_index: bool = False,
    ):
        self._verify_dirs(input_dir=input_dir, data_dir=data_dir)
        self.paragraphs_per_search = 3
        
        # Check if index exists
        if not os.path.exists(data_dir+'/'+CHUNKS_INDEX_NAME) or not os.path.exists(data_dir+'/'+CHUNKS_DF_NAME):
            print("Index does not exist. Building index.")
            
            self._build_index_from_dir(input_dir, data_dir, chunk_size=chunk_size, overlap_chunks=overlap_chunks)
        elif force_rebuild_index:
            
            print("Index exists. Forced rebuilding of index.")
            self._build_index_from_dir(input_dir, data_dir, chunk_size=chunk_size, overlap_chunks=overlap_chunks)
        else:
            
            print("Index already exists. Loading index.")
        
        try:
            self.index_chunks = read_index(self.data_dir+'/'+CHUNKS_INDEX_NAME)
            
            self.df_corpus = pd.read_csv(self.data_dir+'/'+CHUNKS_DF_NAME)
            
        except Exception as e:
            raise Exception(f"Could not load chunks index. Verify if folder {input_dir} contains data.") from e
        
    
    def init_json_retriever(
        self,
        json_file_path: str, 
        data_dir: str = "retriever_data", 
        json_key_path: list[str] = ["area_layout", "offices"],
        json_indexing_keys: list[str] = ["name"],
        force_rebuild_index: bool = False,
    ):
        self._verify_dirs(input_dir=json_file_path, data_dir=data_dir)

        # Check if index exists
        if not os.path.exists(data_dir+'/'+CHUNKS_INDEX_NAME) or not os.path.exists(data_dir+'/'+CHUNKS_DF_NAME):
            print("Index does not exist. Building index.")
            self._build_index_from_json(json_file_path, data_dir, json_key_path=json_key_path, json_indexing_keys=json_indexing_keys)
        elif force_rebuild_index:
            print("Index exists. Forced rebuilding of index.")
            self._build_index_from_json(json_file_path, data_dir, json_key_path=json_key_path, json_indexing_keys=json_indexing_keys)
        else:
            print("Index already exists. Loading index.")

        try:
            self.index_chunks = read_index(self.data_dir+'/'+CHUNKS_INDEX_NAME)
            self.df_corpus = pd.read_csv(self.data_dir+'/'+CHUNKS_DF_NAME)
        except Exception as e:
            raise Exception(f"Could not load chunks index. Verify if folder {json_file_path} contains data.") from e
        
        

    def _verify_dirs(self, input_dir: str, data_dir: str):
        # Creating data_dir if doesn't exist
        if data_dir[-1] == '/':
            data_dir = data_dir[:-1]
        if input_dir[-1] == '/':
            input_dir = input_dir[:-1]
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        self.input_dir = input_dir
        if not os.path.exists(input_dir):
            raise Exception(f"Input directory {input_dir} does not exist. Initialize the retriever with a valid folder with the data.")
        

    def _build_index_from_json(
        self,
        json_file_path: str, 
        data_dir: str = "retriever_data", 
        json_key_path: list[str] = ["area_layout", "offices"],
        json_indexing_keys: list[str] = ["name"],
    ):
        """Builds the index from a json file."""
        df_chunks_json = create_df_corpus_from_json(json_file_path, json_key_path=json_key_path, json_indexing_keys=json_indexing_keys)
        # Save df_chunks_json to csv
        df_chunks_json.to_csv(data_dir+'/'+CHUNKS_DF_NAME, index=False, escapechar="\\")
        create_index(
            df_corpus_path_dir=data_dir,
            df_corpus_name=CHUNKS_DF_NAME,
            encoder_model=self.model,
            index_column = "reference",
            index_dir = data_dir,
            index_name = CHUNKS_INDEX_NAME,
            encoder_model_dimensions=self.model.get_sentence_embedding_dimension(),
            verbose = True
        )

    def _build_index_from_dir(self, input_dir: str, data_dir: str, chunk_size: str = 1000, overlap_chunks: str = 0):
        """Builds the index from the input directory.
        
        If there is pdf files, convert to .txt;
        Divide txt files into chunks of chunk_size and overlap;
        """
        # Convert pdf files to txt
        pdf_to_text(input_dir, verbose = True)
        # Create df_corpus with txt files
        df_chunks_txt = create_df_corpus_from_txt(texts_path_dir=input_dir, paragraph_size=chunk_size, paragraph_overlap=overlap_chunks, verbose = True)
        # Save df_chunks_txt to csv
        df_chunks_txt.to_csv(data_dir+'/'+CHUNKS_DF_NAME, index=False, escapechar="\\")
        create_index(
            df_corpus_path_dir=data_dir,
            df_corpus_name=CHUNKS_DF_NAME,
            encoder_model=self.model,
            index_column = "text",
            index_dir = data_dir,
            index_name = CHUNKS_INDEX_NAME,
            encoder_model_dimensions=self.model.get_sentence_embedding_dimension(),
            verbose = True
        )
        

    def _fetch_text(self, dataframe_idx):
        def merge_strings(str1, str2):
            str1 = str(str1)
            str2 = str(str2)
            # Find the common overlap
            overlap = find_overlap(str1, str2)
            # Concatenate the strings, excluding the common overlap
            return str1 + str2[len(overlap):]
        def find_overlap(str1, str2):
            max_overlap = min(len(str1), len(str2))
            overlap = ""
            # Iterate through possible overlaps
            for i in range(1, max_overlap + 1):
                if str1.endswith(str2[:i]):
                    overlap = str2[:i]
            return overlap
        if(self.paragraphs_per_search == 1):
            info = self.df_corpus.iloc[dataframe_idx]
            return info.text
        else:
            info = self.df_corpus.iloc[dataframe_idx-(int(self.paragraphs_per_search/2)):dataframe_idx+int(self.paragraphs_per_search/2)+1]
        text = ""
        for info1 in info.text:
            text = merge_strings(text, info1)
        return text
        
    def _search_query(self, query, top_k, previous_context = [], verbose = False):
        t=time.time()
        results = []
        query_vector = self.model.encode([query])

        top_k_results_text = self.index_chunks.search(query_vector, 3*top_k)     
        #Process results from raw text
        top_k_ids_text = unique(top_k_results_text[1].tolist()[0])
        #Verify if top_k_ids_text have indexes close to each other
        for i in range(len(top_k_ids_text)):
            for j in range(0, i):
                if abs(top_k_ids_text[i] - top_k_ids_text[j]) <= self.paragraphs_per_search:
                    top_k_ids_text[i] = top_k_ids_text[j]
        #Verify if the context is in previous_context
        i = 0
        while i < len(top_k_ids_text):
            text = self._fetch_text(top_k_ids_text[i])
            if text in previous_context:
                top_k_ids_text = np.delete(top_k_ids_text, i)
            else:
                i += 1
        #clean top_k_ids_text 
        top_k_ids_text = unique(top_k_ids_text)[:top_k]
        results_text = [self._fetch_text(idx) for idx in top_k_ids_text]

        results += results_text

        if verbose:
            print('>>>> Results in Total Time: {}'.format(time.time()-t))

        return results


    def get_context(
            self, 
            query: str,
            top_k: int = 3,
            verbose = False):
        """Expands the original query and return relevant context."""
        return self._search_query(query, top_k, previous_context = [], verbose = verbose)



if __name__ == "__main__":
    retriever = SentenceRetriever()
    while(1):
        query = str(input("Query: "))
        #What are the main ingredients for a blast furnace?
        results = retriever.search_query(query, 2)
        for result in results:
            print("\n\n---------\n\n")
            print(result) 





