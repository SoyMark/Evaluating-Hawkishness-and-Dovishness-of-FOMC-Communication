import os
from pathlib import Path
from typing import List, Tuple
import nltk
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from datetime import datetime
import sys
from tqdm import tqdm

# Parameters
DATA_DIR = Path(sys.argv[1])
# DATA_DIR = Path("corpus/fed_speeches")
# DATA_DIR = Path("corpus/fomc_minutes")
# DATA_DIR = Path("corpus/press_conferences")
BATCH_SIZE = 16

# Load tokenizer and model
model_name = "gtfintechlab/FOMC-RoBERTa"
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True, do_basic_tokenize=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Sentence splitting
nltk.download("punkt")
def split_sentences(text: str) -> List[str]:
    return nltk.sent_tokenize(text)

# Load all (sentence, source_file) pairs
def load_all_sentences(directory: Path) -> List[Tuple[str, str]]:
    all_entries = []
    for file in sorted(directory.glob("*.txt")):
        with file.open("r", encoding="utf-8") as f:
            text = f.read()
        sentences = split_sentences(text)
        for sentence in sentences:
            text = re.sub(r'[\r\n]+', '', text)
            all_entries.append((file.name, sentence))
    return all_entries

# Inference function
def classify_batch(sentences: List[str]) -> List[List[float]]:
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
    return probs.cpu().tolist()

# extract date from document name
def extract_date(doc_name):
    raw_date = doc_name.split('.')[0]
    return datetime.strptime(raw_date, "%Y%m%d").strftime("%Y-%m-%d")

# Main routine
def main():
    entries = load_all_sentences(DATA_DIR)
    print(f"Total sentences: {len(entries)}")

    all_results = []
    for i in tqdm(range(0, len(entries), BATCH_SIZE)):
        batch_entries = entries[i:i+BATCH_SIZE]
        filenames, sentences = zip(*batch_entries)
        prob_list = classify_batch(list(sentences))
        for doc, sent, probs in zip(filenames, sentences, prob_list):
            all_results.append({
                "document": doc,
                "sentence": sent,
                "dovish": probs[0],
                "hawkish": probs[1],
                "neutral": probs[2],
            })

    df = pd.DataFrame(all_results)
    df.to_csv(f"result/{sys.argv[1].split('/')[-1]}/fomc_sen_classification.csv", index=False)

    agg_df = df.groupby("document")[["dovish", "hawkish", "neutral"]].mean().reset_index()
    agg_df["date"] = agg_df["document"].apply(extract_date)
    agg_df = agg_df.sort_values(by="date")
    output_file = f"result/{sys.argv[1].split('/')[-1]}/classification_vector.csv"
    agg_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
