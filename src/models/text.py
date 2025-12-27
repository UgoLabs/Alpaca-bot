import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig

class TextHead(nn.Module):
    """
    Processes financial news/sentiment using a lightweight DistilBERT.
    """
    def __init__(self, output_dim=128, freeze_bert=True):
        super(TextHead, self).__init__()
        
        # Load pre-trained DistilBERT
        # We use a config to avoid downloading weights if we want to initialize from scratch,
        # but for a real bot, we'd want 'distilbert-base-uncased-finetuned-sst-2-english' or similar.
        # Here we assume standard 'distilbert-base-uncased' structure.
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
        self.fc = nn.Linear(768, output_dim) # DistilBERT hidden size is 768
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        # input_ids: (Batch, Seq_Len)
        # attention_mask: (Batch, Seq_Len)
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Take the [CLS] token embedding (first token) as the sentence summary
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        out = self.fc(cls_token)
        out = self.relu(out)
        return out
