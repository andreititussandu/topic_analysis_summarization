from transformers import pipeline
import torch

# Check if MPS (Apple's GPU) is available and set the device accordingly
#device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

model_name = "facebook/bart-large-cnn"
model_revision = "main"

summarizer = pipeline("summarization", model=model_name, revision=model_revision)

def summarize_text(text):
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']
