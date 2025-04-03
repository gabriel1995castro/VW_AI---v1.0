import ollama
import numpy as np

class OllamaEmbedding:
    def __init__(self, encoder_model_id: str):
        self.encoder_model_id = encoder_model_id
        ollama.pull(self.encoder_model_id)
        show = ollama.show(self.encoder_model_id)
        model_info = show["model_info"]
        for key in model_info.keys():
            if "embedding_length" in key:
                self.context_length = model_info[key]
                print("embedding_length: ", self.context_length)
        if not self.context_length:
            self.context_length = 2048


    def get_sentence_embedding_dimension(self):
        return self.context_length

    def encode(self, sentences: list[str], show_progress_bar=False):
        result = []
        for sentence in sentences:
            response = ollama.embeddings(model=self.encoder_model_id, prompt=sentence)
            embedding = response["embedding"]
            result.append(embedding)
        return np.asarray(result)
