from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModel, AutoTokenizer

class TextEncoder(torch.nn.Module):
    def __init__(self, config):
        super(TextEncoder, self).__init__()
        self.encoder_name = config["text_encoder_name"]
        try:
            self.model = SentenceTransformer(self.encoder_name)
            self.d_model = self.model.get_sentence_embedding_dimension()
            self.use_sentence_transformer = True
        except Exception:
            self.model = AutoModel.from_pretrained(self.encoder_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
            self.d_model = self.model.config.hidden_size
            self.use_sentence_transformer = False
    def forward(self, batch_input):
        '''
        Input:
            batch_input: list of dicts
                each dict contains:
                    - global: str
                    - channel_1: str
                    - channel_2: str
                    - ...
        Output:
            batch_channel_embeddings: torch.Tensor
                shape: (B, C, d_model)
            batch_global_embeddings: torch.Tensor
                shape: (B, d_model)
        '''
        batch_size = len(batch_input)
        num_channels = len(batch_input[0]) - 1  # exclude 'global'

        global_sentences = [input_dict["global"] for input_dict in batch_input]
        channel_sentences = []
        for input_dict in batch_input:
            for key in input_dict:
                if key != "global":
                    channel_sentences.append(input_dict[key])
        
        if self.use_sentence_transformer:
            channel_emb = self.model.encode(channel_sentences, convert_to_tensor=True)  # [B*C, d_model]
            global_emb = self.model.encode(global_sentences, convert_to_tensor=True)  # [B，d_model]
        else:
            channel_inputs = self.tokenizer(channel_sentences, padding=True, truncation=True, return_tensors="pt")
            global_input = self.tokenizer(global_sentences, padding=True, truncation=True, return_tensors="pt")

            with torch.no_grad():
                channel_outputs = self.model(**channel_inputs)
                global_outputs = self.model(**global_input)

            global_emb = global_outputs.last_hidden_state.mean(dim=1)  # [B, d_model]
            channel_emb = channel_outputs.last_hidden_state.mean(dim=1).view(batch_size, num_channels, -1)  # [B, C, d_model]
            
        return {"channel": channel_emb, "global": global_emb}



def get_text_encoder_dimension(text_encoder_name):
    """
    Get the dimension of the hidden embedding (d_model) given the encoder name
    
    Args:
        text_encoder_name: The name of the text encoder model
        
    Returns:
        d_model: The dimension of the hidden embedding
    """
    # Common model dimensions
    model_dimensions = {
        "sentence-transformers/all-mpnet-base-v2": 768,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
        "bert-base-uncased": 768,
        "bert-large-uncased": 1024,
        "roberta-base": 768,
        "roberta-large": 1024,
        "distilbert-base-uncased": 768,
        "distilroberta-base": 768,
        "albert-base-v2": 768,
        "albert-large-v2": 1024,
        "google/electra-base-discriminator": 768,
        "google/electra-large-discriminator": 1024,
        "nomic-ai/nomic-embed-text-v1": 768,
        "nomic-ai/nomic-embed-text-v1.5": 768,
        "intfloat/e5-base-v2": 768,
        "intfloat/e5-large-v2": 1024,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024
    }
    
    if text_encoder_name in model_dimensions:
        return model_dimensions[text_encoder_name]
    
    try:
        try:
            temp_model = SentenceTransformer(text_encoder_name)
            return temp_model.get_sentence_embedding_dimension()
        except Exception:
            temp_model = AutoModel.from_pretrained(text_encoder_name)
            return temp_model.config.hidden_size
    except Exception as e:
        print(f"Error determining dimension for {text_encoder_name}: {e}")
        return 768

