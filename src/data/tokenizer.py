from transformers import DistilBertTokenizer
import torch

class FinancialTokenizer:
    def __init__(self, max_len=32):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.max_len = max_len

    def tokenize(self, texts):
        """
        Tokenizes a list of text strings.
        Returns (input_ids, attention_mask) tensors.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        encoding = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return encoding['input_ids'], encoding['attention_mask']
