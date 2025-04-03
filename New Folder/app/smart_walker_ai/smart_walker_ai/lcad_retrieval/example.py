"""Example usage of the SentenceRetriever class."""

from lcad_retriever import SentenceRetriever
import os
hf_token = os.environ["HF_TOKEN"]

if __name__ == "__main__":
    retriever = SentenceRetriever(hf_token=hf_token)
    retriever.init_json_retriever(
        json_file_path="../data/general_map_description.json", 
        json_key_path=["area_layout", "offices"],
        json_indexing_keys=["name"],
        force_rebuild_index=True
    )

    # Test the retriever
    query = "I would like to go somewhere to eat"
    print("\n\n>>Query: ", query)
    results = retriever.get_context(query, top_k=2)
    for result in results:
        print(result)
        print()

    query = "Where is the HCS?"
    print("\n\n>>Query: ", query)
    results = retriever.get_context(query, top_k=2)
    for result in results:
        print(result)
        print()

    ######
    retriever.init_pdf_retriever(
        input_dir="../texts/ml",
        force_rebuild_index=True
    )

    # Test the retriever
    query = "What is machine learning?"
    print("\n\n>>Query: ", query)
    results = retriever.get_context(query, top_k=2)
    for result in results:
        print(result)
        print()

    query = "What is continuos learning?"
    print("\n\n>>Query: ", query)
    results = retriever.get_context(query, top_k=2)
    for result in results:
        print(result)
        print()