import requests
from lxml import etree
from bs4 import BeautifulSoup
import re

HEADERS = {'User-Agent': 'Mozilla/5.0'}

def get_links_from_sitemap(url):
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        root = etree.fromstring(response.content)
        return root.xpath('//ns:loc/text()', namespaces={'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'})
    except requests.exceptions.RequestException as e:
        print(f"Error fetching sitemap from {url}: {e}")
        return []

def get_muddy_waters_links(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file.read(), 'html.parser')
    links = []
    for row in soup.find_all('tr'):
        first_td = row.find('td', class_='first')
        last_td = row.find('td', class_='last')
        if first_td and last_td:
            a_tag = first_td.find('a')
            if a_tag:
                links.append({
                    'url': a_tag['href'],
                    'title': a_tag['title'],
                    'date': last_td.text.strip()
                })
    return links

def extract_ticker(url):
    if "muddywatersresearch.com/research/" in url:
        match = re.search(r'/research/([^/]+)/', url)
        if match:
            return match.group(1).upper()
    words = re.sub(r'https?://|www\.|\.com|research|items|hindenburg|wolfpack', '', url).split('/')[-1].split('-')
    for word in words:
        if word.isupper() and 1 <= len(word) <= 5:
            return word
        if word.startswith('$') and 1 <= len(word[1:]) <= 5:
            return word[1:]
    return None

sitemap_urls = [
    "https://hindenburgresearch.com/wp-sitemap-posts-post-1.xml",
    "https://www.wolfpackresearch.com/dynamic-items-sitemap.xml"
]

all_links = [link for url in sitemap_urls for link in get_links_from_sitemap(url)]
muddy_waters_links = get_muddy_waters_links('mwreports.txt')

ticker_dict = {}
for link in all_links:
    if ticker := extract_ticker(link):
        ticker_dict.setdefault(ticker, []).append({'url': link, 'source': 'Hindenburg/Wolfpack'})

for link in muddy_waters_links:
    if ticker := extract_ticker(link['url']):
        ticker_dict.setdefault(ticker, []).append({'url': link['url'], 'title': link['title'], 'date': link['date'], 'source': 'Muddy Waters'})

with open('shortreport.txt', 'w', encoding='utf-8') as file:
    file.write("Extracted tickers and their corresponding URLs:\n")
    for ticker, urls in ticker_dict.items():
        file.write(f"\nTicker: {ticker}\n")
        for item in urls:
            if item['source'] == 'Muddy Waters':
                file.write(f"- {item['url']} (Muddy Waters, {item['date']}: {item['title']})\n")
            else:
                file.write(f"- {item['url']} ({item['source']})\n")

    file.write(f"\nTotal unique tickers found: {len(ticker_dict)}\n")

    file.write("\nHindenburg Research URLs:\n")
    for link in all_links:
        if "hindenburgresearch" in link:
            file.write(f"{link}\n")

    file.write("\nWolfpack Research URLs:\n")
    for link in all_links:
        if "wolfpackresearch" in link:
            file.write(f"{link}\n")

    file.write("\nMuddy Waters Research URLs:\n")
    for link in muddy_waters_links:
        file.write(f"{link['url']} ({link['date']}: {link['title']})\n")
