import functools
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

@functools.cache
def load_deberta_tokenizer_and_model():

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    model = AutoModelForSequenceClassification.from_pretrained(
        "yimingzhang/deberta-v3-large-generation-similarity"
    ).to(DEVICE)
    model.eval()
    return tokenizer, model