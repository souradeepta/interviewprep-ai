# Web Agents
**TL;DR:** A web agent interacts with permitted sites through a resilient browser or documented API, validating page state before each consequential action.

## Prerequisites
Agent loops, tool boundaries, and evaluation.

## Learning Outcomes
- Explain the main web action workflow and its state boundary.
- Compare an automated path with review or fallback using measured outcomes.
- Diagnose a representative failure using trace evidence.

## Core Intuition
Web Agents is best understood as a controlled workflow with explicit evidence and a bounded decision. The system should expose what it knows, what it is allowed to do, and when a person or deterministic checker must take over. This framing prevents a fluent answer from being confused with a verified outcome.

## How It Works
Prefer a documented API where available; browser automation is appropriate when the interface itself is the required surface. Use explicit waits for state changes, validate origin and page identity, and keep credentials in a secret store. Treat page content as untrusted instructions and respect access policy, rate limits, and user authorization.
## Detailed Explanation

Web agents interact with websites through browser automation or API calls. Core mechanisms: (1) browser automation—control browser to click, type, navigate (Selenium, Puppeteer), (2) API calls—direct HTTP requests to web services, (3) parsing—extract data from HTML (BeautifulSoup, CSS selectors). Advantages: automate repetitive web tasks (data scraping, form filling), access web services not designed for automation, test web applications. Challenges: website changes break automation (selectors become invalid), JavaScript rendering (agent must wait for page to load), handling authentication (cookies, sessions), rate limiting (don't overwhelm server), legal/ethical concerns (respect robots.txt, terms of service). Best for: data collection (public data from websites), workflow automation (fill forms across multiple sites), testing (automated testing of web apps), content aggregation (collect articles from multiple sources).


## Core Intuition

Imagine hiring someone to browse websites and fill forms for you. They open browser, navigate to site, read instructions on page, click buttons, fill forms, extract data. Web agents are automated web browsing and interaction.


## How It Works

Web agents operate through: navigate → select element → perform action → wait → extract data:

1. **Navigation** — Open website or call API endpoint
2. **Element Selection** — Find button/input using CSS selector or XPath
3. **Action** — Click button, type text, submit form, or call API
4. **Wait** — Wait for page load or response
5. **Extraction** — Parse HTML/JSON, extract needed data
6. **Validation** — Verify action succeeded
7. **Next Step** — Repeat or finish




## Interview Q&A

**Q: When should you use browser automation vs API calls?**
A: API if available (faster, reliable, respects service). Browser automation if no API (web scraping, web app testing, JavaScript-dependent). Hybrid: try API first, fallback to browser.

**Q: How do you handle website changes that break selectors?**
A: (1) Use stable selectors (ID > class > XPath), (2) Monitor selectors (test regularly), (3) Fallback selectors (multiple ways to find same element), (4) Alerts (notify when selector fails).

**Q: How do you avoid overwhelming servers?**
A: Add delays between requests (random jitter), respect robots.txt, check rate limit headers, use API instead of scraping when available.

**Q: How do you handle JavaScript-heavy sites?**
A: Use browser automation (Selenium, Puppeteer) that executes JavaScript. Wait for dynamic content to load before extracting.

**Q: Is web scraping legal?**
A: Gray area. Check: (1) terms of service (usually prohibit), (2) robots.txt (instructions for bots), (3) legality (varies by jurisdiction), (4) data sensitivity (public vs private). Better: use official API.

**Q: How do you maintain web agent as websites evolve?**
A: (1) Monitoring—alert when selectors break, (2) Testing—regular tests with real site, (3) Flexible selectors—multiple ways to find elements, (4) Documentation—why this selector was chosen.
Measure task completion, selector drift, retry rate, blocked navigation, latency, and duplicate submissions. Test changed layouts, slow rendering, login expiry, and unexpected interstitial pages.

## Trade-offs
| Choice | Benefit | Cost or boundary |
|---|---|---|
| Deterministic baseline | Predictable behavior and low cost | Limited coverage of ambiguous cases |
| Model-assisted path | Handles flexible inputs | Requires evidence checks and evaluation |
| Human review | Resolves high-impact uncertainty | Capacity and response-time limits |
| Fallback or abstention | Bounds unsafe behavior | May defer useful work |

## Implementation
This runnable Python example shows a bounded policy gate relevant to the workflow. It models control logic only; it does not claim production quality.

```python
from urllib.parse import urlparse
def allowed_url(url, domains):
    return urlparse(url).hostname in domains

assert allowed_url('https://docs.example.org/page', {'docs.example.org'})
assert not allowed_url('https://evil.test/', {'docs.example.org'})
```

## Failure Modes
- Web Agents uses stale or incomplete input: log source and timestamp, validate freshness, and request a refresh before acting.
- Web Agents produces an unsupported result: require evidence and a domain-specific validator; route unresolved cases to review.
- Web Agents exceeds its resource or authority boundary: enforce explicit budgets and permissions in the controller, then report the stopped state.

## Interview Quick Reference
| Question | Expected answer and follow-up |
|---|---|
| When should you use browser automation vs API calls? | API if available (faster, reliable, respects service). Browser automation if no API (web scraping, web app testing, JavaScript-dependent). Hybrid: try API first, fallback to browser. |
| How do you handle website changes that break selectors? | (1) Use stable selectors (ID > class > XPath), (2) Monitor selectors (test regularly), (3) Fallback selectors (multiple ways to find same element), (4) Alerts (notify when selector fails). |
| How do you avoid overwhelming servers? | Add delays between requests (random jitter), respect robots.txt, check rate limit headers, use API instead of scraping when available. |

## Related Concepts
See [agent testing](28-agent-testing.md), [agent evaluation](29-agent-evals.md), and [agent monitoring](30-agent-monitoring.md).

## References
- [playwright.dev](https://playwright.dev/docs/intro)
- [www.w3.org](https://www.w3.org/TR/webdriver2/)

## Further Topic Detail
## Detailed Explanation

Web agents interact with websites through browser automation or API calls. Core mechanisms: (1) browser automation—control browser to click, type, navigate (Selenium, Puppeteer), (2) API calls—direct HTTP requests to web services, (3) parsing—extract data from HTML (BeautifulSoup, CSS selectors). Advantages: automate repetitive web tasks (data scraping, form filling), access web services not designed for automation, test web applications. Challenges: website changes break automation (selectors become invalid), JavaScript rendering (agent must wait for page to load), handling authentication (cookies, sessions), rate limiting (don't overwhelm server), legal/ethical concerns (respect robots.txt, terms of service). Best for: data collection (public data from websites), workflow automation (fill forms across multiple sites), testing (automated testing of web apps), content aggregation (collect articles from multiple sources).


## Core Intuition

Imagine hiring someone to browse websites and fill forms for you. They open browser, navigate to site, read instructions on page, click buttons, fill forms, extract data. Web agents are automated web browsing and interaction.


## How It Works

Web agents operate through: navigate → select element → perform action → wait → extract data:

1. **Navigation** — Open website or call API endpoint
2. **Element Selection** — Find button/input using CSS selector or XPath
3. **Action** — Click button, type text, submit form, or call API
4. **Wait** — Wait for page load or response
5. **Extraction** — Parse HTML/JSON, extract needed data
6. **Validation** — Verify action succeeded
7. **Next Step** — Repeat or finish




## Interview Q&A

**Q: When should you use browser automation vs API calls?**
A: API if available (faster, reliable, respects service). Browser automation if no API (web scraping, web app testing, JavaScript-dependent). Hybrid: try API first, fallback to browser.

**Q: How do you handle website changes that break selectors?**
A: (1) Use stable selectors (ID > class > XPath), (2) Monitor selectors (test regularly), (3) Fallback selectors (multiple ways to find same element), (4) Alerts (notify when selector fails).

**Q: How do you avoid overwhelming servers?**
A: Add delays between requests (random jitter), respect robots.txt, check rate limit headers, use API instead of scraping when available.

**Q: How do you handle JavaScript-heavy sites?**
A: Use browser automation (Selenium, Puppeteer) that executes JavaScript. Wait for dynamic content to load before extracting.

**Q: Is web scraping legal?**
A: Gray area. Check: (1) terms of service (usually prohibit), (2) robots.txt (instructions for bots), (3) legality (varies by jurisdiction), (4) data sensitivity (public vs private). Better: use official API.

**Q: How do you maintain web agent as websites evolve?**
A: (1) Monitoring—alert when selectors break, (2) Testing—regular tests with real site, (3) Flexible selectors—multiple ways to find elements, (4) Documentation—why this selector was chosen.
