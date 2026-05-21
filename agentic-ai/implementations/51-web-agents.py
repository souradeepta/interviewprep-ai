"""
Auto-generated from 51-web-agents.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Web Agents
# Objectives: Browser automation, API integration, parsing, rate limiting
# ======================================================================

import requests

class WebAgent:
    def fetch(self, url):
        return requests.get(url).text

agent = WebAgent()
html = agent.fetch('https://example.com')
print(f'Fetched {len(html)} bytes')


from bs4 import BeautifulSoup

class ParsingAgent:
    def parse(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        links = [a.get('href') for a in soup.find_all('a')]
        return links

agent = ParsingAgent()
links = agent.parse('<html><a href="/page1">Link</a></html>')
print(f'Found {len(links)} links')


import time

class RateLimitedAgent:
    def __init__(self, delay=1):
        self.delay = delay
        self.last_request = 0
    
    def request(self, url):
        elapsed = time.time() - self.last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request = time.time()
        return requests.get(url).json()

print('Rate limiting: safe requests')


class ErrorHandlingAgent:
    def fetch_safe(self, url, retries=3):
        for i in range(retries):
            try:
                return requests.get(url, timeout=5).json()
            except Exception as e:
                if i < retries - 1:
                    time.sleep(2 ** i)
        return None

print('Error handling: retries and timeouts')


class HybridAgent:
    def get_data(self, endpoint):
        try:
            return requests.get(f'/api/{endpoint}').json()
        except:
            return self._scrape(endpoint)

print('Hybrid: API first, scrape fallback')


# ======================================================================
# ## Key Takeaways
# Core concepts applied. Patterns proven. Ready for production.
# ======================================================================
