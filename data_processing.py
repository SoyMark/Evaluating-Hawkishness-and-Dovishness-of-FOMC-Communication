import pandas as pd
import re
from datetime import datetime

def extract_date(doc_name):
    match = re.search(r"(\d{8})", doc_name)
    if match:
        raw_date = match.group(1)
        return datetime.strptime(raw_date, "%Y%m%d").strftime("%Y-%m-%d")
    else:
        return None

# get 90-day Treasury/5-year yield spread
def process_treasury_data():
    
    dgs5 = pd.read_csv("data/DGS5.csv", parse_dates=["observation_date"])
    dtb3 = pd.read_csv("data/DTB3.csv", parse_dates=["observation_date"])

    dgs5.rename(columns={"observation_date": "date", "DGS5": "DGS5"}, inplace=True)
    dtb3.rename(columns={"observation_date": "date", "DTB3": "DTB3"}, inplace=True)

    merged = pd.merge(dtb3, dgs5, on="date", how="inner")

    # Compute the yield spread: 90-day Treasury yield minus 5-year Treasury yield
    merged["90-day Treasury/5-year yield spread"] = merged["DTB3"] - merged["DGS5"]

    result = merged[["date", "90-day Treasury/5-year yield spread"]]
    result.to_csv("data/treasury_spread.csv", index=False, float_format="%.6f")

def aggregate_classification_result():
    input_file = "result/fomc_classification.csv" 
    df = pd.read_csv(input_file)
    agg_df = df.groupby("document")[["dovish", "hawkish", "neutral"]].mean().reset_index()
    
    agg_df["date"] = agg_df["document"].apply(extract_date)

    agg_df = agg_df.sort_values(by="date")
    output_file = "result/classification_vector.csv"
    agg_df.to_csv(output_file, index=False)
    
def aggregate_similarity_result():
    input_file = "result/finbert_prosus_document_vector.csv" 
    df = pd.read_csv(input_file)
    
    df["date"] = df["document"].apply(extract_date)

    df = df.sort_values(by="date")
    output_file = "result/similarity_vector.csv"
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    process_treasury_data()
    # aggregate_classification_result()
    # aggregate_similarity_result()