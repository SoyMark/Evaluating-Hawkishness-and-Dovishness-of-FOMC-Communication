#!/usr/bin/env python3
"""
Download each FOMC meeting *Statement* page and save **only** the body
text (excluding release line & voting paragraph) to plain-text files.
"""
import os
import re
import time
import requests
from bs4 import BeautifulSoup

BASE_URL   = "https://www.federalreserve.gov"
CAL_URL    = f"{BASE_URL}/monetarypolicy/fomccalendars.htm"
OUT_DIR    = "fomc_statement_txt"
DELAY_SECS = 0.1       # polite interval
HEADERS    = {"User-Agent": "Mozilla/5.0 (FOMC-Scraper/1.1)"}

# The FOMC statement body is in <p> tags, but the first <p> tag
def extract_body_text(html: str) -> str:
    """
    Return ONLY the statement body paragraphs from a Fed FOMC statement page.
    """
    soup = BeautifulSoup(html, "html.parser")


    article = soup.find(id="article") or soup.body

    body_paras = []
    for p in article.find_all("p"):
        text = p.get_text(" ", strip=True)
        if not text:                                   
            continue

        if (
            p.get("class") in (["releaseTime"], ["article__time"])  
            or text.lower().startswith("for release")               
            or "implementation note" in text.lower()               
        ):
            continue

        if text.startswith("Voting for"):
            break

        body_paras.append(text)

    return "\n\n".join(body_paras).strip()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Fetching calendar → {CAL_URL}")
    soup = BeautifulSoup(requests.get(CAL_URL, headers=HEADERS, timeout=30).text,
                         "html.parser")

    for label in soup.find_all(string=re.compile(r"^\s*Statement:\s*$")):
        link = label.find_next("a", string="HTML")
        if not link or not link["href"].endswith(".htm"):
            continue           

        url     = link["href"] if link["href"].startswith("http") else BASE_URL + link["href"]
        base    = os.path.basename(link["href"])
        out_txt = os.path.join(OUT_DIR, base.replace(".htm", ".txt"))

        if os.path.exists(out_txt):
            print(f"skip {out_txt}")
            continue

        print(f"↓ Downloading → {url}")
        html = requests.get(url, headers=HEADERS, timeout=30).text
        body = extract_body_text(html)

        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(body)

        time.sleep(DELAY_SECS)

if __name__ == "__main__":
    main()
