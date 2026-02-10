import requests
from bs4 import BeautifulSoup
import pdfkit
import os
from urllib.parse import urljoin

BASE_URL = "https://www.geeksforgeeks.org/machine-learning/machine-learning/"
DOMAIN = "https://www.geeksforgeeks.org"

visited = set()
article_links = []

def get_article_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Collect all internal ML links
    for a in soup.find_all("a", href=True):
        link = a["href"]

        if link.startswith("/machine-learning/"):
            full_link = urljoin(DOMAIN, link)

            if full_link not in visited:
                visited.add(full_link)
                article_links.append(full_link)

# Step 1: Get all tutorial links
print("Collecting links...")
get_article_links(BASE_URL)

print(f"Total pages found: {len(article_links)}")

# Step 2: Convert each page to HTML file
combined_html = "<h1>GeeksforGeeks - Machine Learning Tutorial</h1>"

for link in article_links:
    print(f"Processing: {link}")
    response = requests.get(link)
    soup = BeautifulSoup(response.text, "html.parser")

    article_content = soup.find("article")

    if article_content:
        combined_html += str(article_content)
        combined_html += "<div style='page-break-after: always;'></div>"

# Step 3: Save combined HTML
with open("combined_ml_tutorial.html", "w", encoding="utf-8") as f:
    f.write(combined_html)

print("Generating PDF...")

# Step 4: Convert to PDF
pdfkit.from_file("combined_ml_tutorial.html", "Machine_Learning_Tutorial.pdf")

print("PDF Generated Successfully!")
