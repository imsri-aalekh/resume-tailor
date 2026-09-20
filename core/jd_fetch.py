"""
Fetch a job description from a URL.

Reality check: LinkedIn and Naukri actively block server-side scraping, and a
Streamlit Cloud IP is a datacenter IP, which is exactly what they block. So this
module is built as a ladder of strategies that degrades honestly:

  1. site-specific guest/JSON endpoints (LinkedIn, Naukri, Greenhouse, Lever)
  2. schema.org JobPosting JSON-LD (most modern ATS-hosted boards emit this)
  3. generic main-content extraction
  4. give up loudly and ask the user to paste the text

Never silently return a login wall as if it were a job description — a garbage
JD produces a confidently wrong resume, which is worse than no resume.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")

HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}

TIMEOUT = 25

# Pages that are really a login/verification wall rather than a JD
_WALL_MARKERS = [
    "sign in to continue", "join linkedin", "authwall", "please verify you are a human",
    "enable javascript", "captcha", "access denied", "are you a robot",
    "log in to see", "sign up to see who", "unusual traffic",
]


@dataclass
class JobPosting:
    url: str = ""
    title: str = ""
    company: str = ""
    location: str = ""
    employment_type: str = ""
    seniority: str = ""
    text: str = ""
    source: str = ""          # which strategy succeeded
    ok: bool = False
    error: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def header(self) -> str:
        bits = [b for b in (self.title, self.company, self.location) if b]
        return " | ".join(bits)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _clean(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</(p|li|div|h\d)>", "\n", text, flags=re.I)
    text = re.sub(r"<li[^>]*>", "\n• ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ").replace("​", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(lines).strip()


def _looks_like_wall(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in _WALL_MARKERS)


def _get(url: str, headers: Optional[Dict[str, str]] = None) -> Optional[requests.Response]:
    try:
        r = requests.get(url, headers=headers or HEADERS, timeout=TIMEOUT,
                         allow_redirects=True)
        if r.status_code == 200:
            return r
        return None
    except requests.RequestException:
        return None


# --------------------------------------------------------------------------
# strategy: schema.org JobPosting
# --------------------------------------------------------------------------


def _from_jsonld(soup: BeautifulSoup) -> Optional[Dict]:
    for tag in soup.find_all("script", {"type": "application/ld+json"}):
        raw = tag.string or tag.get_text() or ""
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            # some sites emit several concatenated objects
            try:
                data = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
            except Exception:
                continue
        candidates = data if isinstance(data, list) else [data]
        if isinstance(data, dict) and "@graph" in data:
            candidates = data["@graph"]
        for c in candidates:
            if isinstance(c, dict) and "JobPosting" in str(c.get("@type", "")):
                return c
    return None


def _apply_jsonld(job: JobPosting, node: Dict) -> None:
    job.title = job.title or _clean(str(node.get("title", "")))
    org = node.get("hiringOrganization") or {}
    if isinstance(org, dict):
        job.company = job.company or _clean(str(org.get("name", "")))
    loc = node.get("jobLocation") or {}
    if isinstance(loc, list) and loc:
        loc = loc[0]
    if isinstance(loc, dict):
        addr = loc.get("address") or {}
        if isinstance(addr, dict):
            parts = [addr.get("addressLocality"), addr.get("addressRegion"),
                     addr.get("addressCountry")]
            parts = [str(p) for p in parts if p and isinstance(p, (str, int))]
            job.location = job.location or ", ".join(parts)
    job.employment_type = job.employment_type or _clean(str(node.get("employmentType", "")))
    desc = node.get("description") or ""
    if desc:
        job.text = _clean(str(desc))


# --------------------------------------------------------------------------
# strategy: site-specific
# --------------------------------------------------------------------------


def _linkedin_job_id(url: str) -> Optional[str]:
    m = re.search(r"/jobs/view/(?:[^/]*-)?(\d{6,})", url)
    if m:
        return m.group(1)
    q = parse_qs(urlparse(url).query)
    for key in ("currentJobId", "jobId", "refId"):
        if key in q and q[key] and q[key][0].isdigit():
            return q[key][0]
    m = re.search(r"(\d{9,})", url)
    return m.group(1) if m else None


def _fetch_linkedin(job: JobPosting) -> bool:
    jid = _linkedin_job_id(job.url)
    if not jid:
        return False
    guest = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{jid}"
    r = _get(guest, {**HEADERS, "Accept": "text/html"})
    if r is None or not r.text.strip():
        return False
    soup = BeautifulSoup(r.text, "lxml")
    desc = soup.select_one(".show-more-less-html__markup, .description__text")
    body = _clean(str(desc)) if desc else _clean(r.text)
    if not body or _looks_like_wall(body) or len(body.split()) < 40:
        return False
    title = soup.select_one(".top-card-layout__title, h1")
    company = soup.select_one(".topcard__org-name-link, .top-card-layout__second-subline a")
    loc = soup.select_one(".topcard__flavor--bullet, .top-card-layout__second-subline span")
    job.title = _clean(title.get_text()) if title else ""
    job.company = _clean(company.get_text()) if company else ""
    job.location = _clean(loc.get_text()) if loc else ""
    job.text = body
    job.source = "linkedin-guest-api"
    return True


def _naukri_job_id(url: str) -> Optional[str]:
    m = re.search(r"-(\d{6,})(?:\?|$)", url)
    if m:
        return m.group(1)
    m = re.search(r"jobId=(\d+)", url)
    return m.group(1) if m else None


def _fetch_naukri(job: JobPosting) -> bool:
    jid = _naukri_job_id(job.url)
    if not jid:
        return False
    api = f"https://www.naukri.com/jobapi/v4/job/{jid}"
    headers = {
        **HEADERS,
        "appid": "121",
        "systemid": "godrejjobs",
        "Accept": "application/json",
        "Referer": job.url,
    }
    r = _get(api, headers)
    if r is None:
        return False
    try:
        data = r.json()
    except ValueError:
        return False
    node = (data or {}).get("jobDetails") or {}
    body = _clean(str(node.get("description", "")))
    extras = []
    for key in ("keySkills", "jobHighlights"):
        val = node.get(key)
        if isinstance(val, list):
            names = [str(v.get("value", v)) if isinstance(v, dict) else str(v) for v in val]
            if names:
                extras.append(f"Key skills: {', '.join(names)}")
        elif isinstance(val, str) and val:
            extras.append(val)
    if extras:
        body = (body + "\n\n" + "\n".join(extras)).strip()
    if not body or len(body.split()) < 30:
        return False
    job.title = _clean(str(node.get("title", "")))
    comp = node.get("companyDetail") or {}
    job.company = _clean(str(comp.get("name", node.get("companyName", ""))))
    places = node.get("locations") or []
    if isinstance(places, list) and places:
        job.location = ", ".join(
            str(p.get("label", p)) if isinstance(p, dict) else str(p) for p in places
        )
    job.text = body
    job.source = "naukri-jobapi"
    return True


def _fetch_greenhouse(job: JobPosting) -> bool:
    m = re.search(r"boards\.greenhouse\.io/([^/]+)/jobs/(\d+)", job.url) or \
        re.search(r"job-boards\.greenhouse\.io/([^/]+)/jobs/(\d+)", job.url)
    if not m:
        return False
    board, jid = m.group(1), m.group(2)
    api = f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs/{jid}"
    r = _get(api, {**HEADERS, "Accept": "application/json"})
    if r is None:
        return False
    try:
        data = r.json()
    except ValueError:
        return False
    job.title = _clean(str(data.get("title", "")))
    job.company = board.replace("-", " ").title()
    loc = data.get("location") or {}
    job.location = _clean(str(loc.get("name", ""))) if isinstance(loc, dict) else ""
    job.text = _clean(str(data.get("content", "")))
    job.source = "greenhouse-api"
    return bool(job.text)


def _fetch_lever(job: JobPosting) -> bool:
    m = re.search(r"jobs\.lever\.co/([^/]+)/([0-9a-f-]{8,})", job.url)
    if not m:
        return False
    company, jid = m.group(1), m.group(2)
    api = f"https://api.lever.co/v0/postings/{company}/{jid}"
    r = _get(api, {**HEADERS, "Accept": "application/json"})
    if r is None:
        return False
    try:
        data = r.json()
    except ValueError:
        return False
    job.title = _clean(str(data.get("text", "")))
    job.company = company.replace("-", " ").title()
    cats = data.get("categories") or {}
    job.location = _clean(str(cats.get("location", "")))
    lists = data.get("lists") or []
    extra = "\n\n".join(
        f"{_clean(str(l.get('text','')))}\n{_clean(str(l.get('content','')))}" for l in lists
    )
    job.text = (_clean(str(data.get("description", ""))) + "\n\n" + extra).strip()
    job.source = "lever-api"
    return bool(job.text)


# --------------------------------------------------------------------------
# strategy: generic page
# --------------------------------------------------------------------------

_CONTENT_SELECTORS = [
    "div.jobsearch-JobComponent-description", "div#jobDescriptionText",
    "div.job-description", "section.job-description", "div[class*='jobDescription']",
    "div[class*='job-details']", "div[data-automation-id='jobPostingDescription']",
    "div[class*='description']", "article", "main", "div#content",
]


def _fetch_generic(job: JobPosting) -> bool:
    r = _get(job.url)
    if r is None:
        job.notes.append("The site refused the request (non-200 response).")
        return False
    soup = BeautifulSoup(r.text, "lxml")

    node = _from_jsonld(soup)
    if node:
        _apply_jsonld(job, node)
        if job.text and len(job.text.split()) >= 40 and not _looks_like_wall(job.text):
            job.source = "schema.org JobPosting"
            return True

    for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "svg", "form"]):
        tag.decompose()

    if not job.title:
        h1 = soup.find("h1")
        job.title = _clean(h1.get_text()) if h1 else _clean(soup.title.get_text() if soup.title else "")

    best, best_len = "", 0
    for sel in _CONTENT_SELECTORS:
        for el in soup.select(sel):
            txt = _clean(el.get_text("\n"))
            if len(txt) > best_len:
                best, best_len = txt, len(txt)
        if best_len > 1200:
            break
    if best_len < 400:
        best = _clean(soup.get_text("\n"))

    job.text = best
    job.source = "generic-html"
    return bool(best) and len(best.split()) >= 50 and not _looks_like_wall(best)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

_PASTE_HINT = (
    "Couldn't read that posting automatically. {site} blocks automated requests "
    "from cloud servers, which is what this app runs on. Open the posting in your "
    "browser, copy the job description text, and paste it into the box below — "
    "everything downstream works identically."
)


def fetch(url_or_text: str) -> JobPosting:
    """Fetch a JD. Accepts a URL or raw pasted text."""
    raw = (url_or_text or "").strip()
    if not raw:
        return JobPosting(error="No job description provided.", ok=False)

    is_url = bool(re.match(r"^https?://", raw)) and len(raw.split()) == 1
    if not is_url:
        job = JobPosting(url="", text=_clean(raw), source="pasted", ok=True)
        first = next((l for l in job.text.splitlines() if l.strip()), "")
        job.title = first[:120]
        if job.word_count < 25:
            job.ok = False
            job.error = "That text is too short to be a job description."
        return job

    job = JobPosting(url=raw)
    host = (urlparse(raw).hostname or "").lower()

    strategies = []
    if "linkedin." in host:
        strategies = [_fetch_linkedin]
    elif "naukri." in host:
        strategies = [_fetch_naukri]
    elif "greenhouse.io" in host:
        strategies = [_fetch_greenhouse]
    elif "lever.co" in host:
        strategies = [_fetch_lever]
    strategies.append(_fetch_generic)

    for strat in strategies:
        try:
            if strat(job):
                job.ok = True
                job.text = _clean(job.text)
                return job
        except Exception as exc:  # a broken site should not crash the app
            job.notes.append(f"{strat.__name__}: {type(exc).__name__}")

    site = "LinkedIn" if "linkedin." in host else \
           "Naukri" if "naukri." in host else "That site"
    job.ok = False
    job.error = _PASTE_HINT.format(site=site)
    return job
