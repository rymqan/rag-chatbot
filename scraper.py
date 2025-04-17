import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import json
import time

from config.config import RAW_DATA_PATH, TARGET_URL, BLACKLIST

MAX_CRAWL = 100
REQUEST_DELAY = 1

urls_to_visit = [TARGET_URL]
visited_urls = set()
scraped_data = []

def is_blacklisted(url: str) -> bool:
    return any(url.startswith(bad) for bad in BLACKLIST)

def clean_text(text: str) -> str:
    return " ".join(text.split())

def get_title_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return "home"
    return path.split("/")[-1]

def is_same_domain(url: str) -> bool:
    """
    Checks if the URL belongs to the same domain or a subdomain of the target domain.
    """
    target_netloc = urlparse(TARGET_URL).netloc
    netloc = urlparse(url).netloc
    return netloc == target_netloc or netloc.endswith(f".{target_netloc}")

def crawl_site():
    crawl_count = 0
    while urls_to_visit and crawl_count < MAX_CRAWL:
        current_url = urls_to_visit.pop(0)
        if current_url in visited_urls or is_blacklisted(current_url):
            continue

        print(f"Crawling: {current_url}")
        try:
            response = requests.get(current_url, timeout=10)
            response.raise_for_status()
        except Exception as e:
            print(f"Error fetching {current_url}: {e}")
            continue

        visited_urls.add(current_url)
        soup = BeautifulSoup(response.text, "html.parser")
        page_text = clean_text(soup.get_text(separator=" ", strip=True))

        scraped_data.append({
            "title": get_title_from_url(current_url),
            "url": current_url,
            "text": page_text
        })

        for link in soup.find_all("a", href=True):
            href = link["href"]
            absolute_url = urljoin(current_url, href)
            if is_same_domain(absolute_url):
                absolute_url = absolute_url.split("#")[0]
                if (absolute_url not in visited_urls and
                        not is_blacklisted(absolute_url) and
                        absolute_url not in urls_to_visit):
                    urls_to_visit.append(absolute_url)

        crawl_count += 1
        time.sleep(REQUEST_DELAY)

def save_json_array():
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    output_path = f"{RAW_DATA_PATH}/raw.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scraped_data, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    crawl_site()
    save_json_array()
