import os
from pathlib import Path
from typing import List, Tuple
import nltk
import torch
import pandas as pd
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
import re
from datetime import datetime
import sys
from tqdm import tqdm

# Parameters
DATA_DIR = Path(sys.argv[1])
# DATA_DIR = Path("corpus/fomc_minutes")
# DATA_DIR = Path("corpus/press_conferences")
BATCH_SIZE = 16
MODEL_NAME = "ProsusAI/finbert"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()
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
            all_entries.append((file.name, sentence))
    return all_entries

# Encoding function
def get_sentence_embeddings(sentences: List[str]) -> torch.Tensor:
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls_embeddings = F.normalize(cls_embeddings, p=2, dim=1)  # L2 normalization
    return cls_embeddings.cpu()

def extract_date(doc_name):
    raw_date = doc_name.split('.')[0]
    return datetime.strptime(raw_date, "%Y%m%d").strftime("%Y-%m-%d")

# Main routine
def main():
    entries = load_all_sentences(DATA_DIR)
    print(f"Total sentences: {len(entries)}")

    # typical dovish sentences and hawkish sentences recommended by chatgpt(who has more finance knowledge than me)
    dovish_sents = [
        "The Committee will be patient as it determines future adjustments to the target range.",
        "The stance of monetary policy remains accommodative.",
        "The Committee is prepared to adjust policy as appropriate to sustain the expansion.",
        "Inflation continues to run below the Committee's symmetric 2 percent objective.",
        "Economic conditions warrant maintaining a lower target range for the federal funds rate."
    ]

    hawkish_sents = [
        "The Committee expects that further gradual increases in the target range will be appropriate.",
        "Labor market conditions remain strong, supporting a more restrictive policy stance.",
        "Inflation has moved close to 2 percent and is expected to rise modestly above that level.",
        "The Committee sees upside risks to inflation and economic overheating.",
        "Monetary policy will need to tighten further to keep inflation expectations anchored."
    ]
    dovish_emb = get_sentence_embeddings(dovish_sents).mean(dim=0)
    hawkish_emb = get_sentence_embeddings(hawkish_sents).mean(dim=0)

    reference_embeddings = torch.stack([dovish_emb, hawkish_emb])

    sentence_results = []
    document_scores = defaultdict(lambda: {"dovish_scores": [], "hawkish_scores": []})

    # Process each batch of sentences
    for i in tqdm(range(0, len(entries), BATCH_SIZE)):
        batch_entries = entries[i:i+BATCH_SIZE]
        filenames, sentences = zip(*batch_entries)
        sentence_embeddings = get_sentence_embeddings(list(sentences))

        for fname, sent, emb in zip(filenames, sentences, sentence_embeddings):
            sim_dovish = torch.cosine_similarity(emb, reference_embeddings[0], dim=0).item()
            sim_hawkish = torch.cosine_similarity(emb, reference_embeddings[1], dim=0).item()

            # sentence level
            sentence_results.append({
                "document": fname,
                "sentence": sent,
                "cosine_with_dovish": f"{sim_dovish:.6f}",
                "cosine_with_hawkish": f"{sim_hawkish:.6f}"
            })

            # aggregation by document
            document_scores[fname]["dovish_scores"].append(sim_dovish)
            document_scores[fname]["hawkish_scores"].append(sim_hawkish)

    df_sent = pd.DataFrame(sentence_results)
    df_sent.to_csv(f"result/{sys.argv[1].split('/')[-1]}/finbert_sentence_similarity.csv", index=False)
    
    doc_summary = []
    # Aggregate results by document
    for fname, scores in document_scores.items():
        avg_dovish = sum(scores["dovish_scores"]) / len(scores["dovish_scores"])
        avg_hawkish = sum(scores["hawkish_scores"]) / len(scores["hawkish_scores"])
        doc_summary.append({
            "document": fname,
            "avg_cosine_with_dovish": avg_dovish,
            "avg_cosine_with_hawkish": avg_hawkish
        })

    df_doc = pd.DataFrame(doc_summary)
    df_doc.to_csv(f"result/{sys.argv[1].split('/')[-1]}/finbert_document_vector.csv", index=False)
    
    df_doc["date"] = df_doc["document"].apply(extract_date)

    df_doc = df_doc.sort_values(by="date")
    output_file = f"result/{sys.argv[1].split('/')[-1]}/similarity_vector.csv"
    df_doc.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
