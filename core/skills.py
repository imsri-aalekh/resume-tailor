"""
Skill vocabulary, aliasing and evidence detection.

The alias table matters more than it looks: a JD that says "Kafka" and a resume
that says "Apache Kafka" are the same skill to a human and two different strings
to an ATS. We normalise both sides before comparing, and we keep the JD's exact
surface form around so the tailored resume can mirror it verbatim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

# --------------------------------------------------------------------------
# canonical skill -> surface forms that mean the same thing
# --------------------------------------------------------------------------

ALIASES: Dict[str, List[str]] = {
    # languages
    "java": ["java", "core java", "java 8", "java 11", "java 17", "java 21", "j2ee", "jee", "java se"],
    "kotlin": ["kotlin"],
    "python": ["python", "python3", "py"],
    "c++": ["c++", "cpp", "c plus plus"],
    "go": ["go", "golang"],
    "scala": ["scala"],
    "javascript": ["javascript", "js", "es6"],
    "typescript": ["typescript", "ts"],
    "sql": ["sql", "ansi sql", "structured query language"],
    "shell": ["shell", "shell scripting", "bash", "unix shell", "ksh"],
    "perl": ["perl"],
    # frameworks / platforms
    "spring boot": ["spring boot", "springboot", "spring-boot"],
    "spring": ["spring", "spring framework", "spring mvc", "spring core"],
    "spring cloud": ["spring cloud", "spring-cloud"],
    "hibernate": ["hibernate", "jpa", "spring data jpa"],
    "microservices": ["microservices", "microservice", "micro services",
                      "microservice architecture", "service oriented architecture", "soa"],
    "rest api": ["rest api", "restful", "rest apis", "restful api", "restful apis",
                 "rest", "restful web services", "web services"],
    "graphql": ["graphql"],
    "grpc": ["grpc", "protobuf", "protocol buffers"],
    "node.js": ["node.js", "nodejs", "node"],
    "react": ["react", "reactjs", "react.js"],
    "angular": ["angular", "angularjs"],
    # messaging / streaming
    "kafka": ["kafka", "apache kafka", "confluent kafka", "kafka streams", "msk"],
    "rabbitmq": ["rabbitmq", "rabbit mq", "amqp"],
    "activemq": ["activemq", "jms"],
    "sqs": ["sqs", "amazon sqs", "aws sqs"],
    "sns": ["sns", "amazon sns", "aws sns"],
    "kinesis": ["kinesis", "aws kinesis"],
    "pubsub": ["pub/sub", "pubsub", "google pub/sub"],
    # cloud
    "aws": ["aws", "amazon web services"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "azure": ["azure", "microsoft azure"],
    "lambda": ["lambda", "aws lambda", "serverless"],
    "ecs": ["ecs", "aws ecs", "elastic container service", "fargate"],
    "eks": ["eks", "aws eks", "elastic kubernetes service"],
    "s3": ["s3", "aws s3", "amazon s3"],
    "dynamodb": ["dynamodb", "dynamo db", "aws dynamodb"],
    "redshift": ["redshift", "aws redshift", "amazon redshift"],
    "step functions": ["step functions", "aws step functions", "state machine"],
    "cloudwatch": ["cloudwatch", "aws cloudwatch"],
    "databrew": ["databrew", "aws glue databrew", "glue databrew"],
    "glue": ["aws glue", "glue etl"],
    "ec2": ["ec2", "aws ec2"],
    "api gateway": ["api gateway", "aws api gateway"],
    # containers / infra
    "docker": ["docker", "containerization", "containerisation", "containers"],
    "kubernetes": ["kubernetes", "k8s", "eks", "openshift"],
    "terraform": ["terraform", "infrastructure as code", "iac"],
    "jenkins": ["jenkins"],
    "ci/cd": ["ci/cd", "cicd", "ci cd", "continuous integration",
              "continuous delivery", "continuous deployment", "build pipeline"],
    "github actions": ["github actions", "gh actions"],
    "helm": ["helm", "helm charts"],
    "linux": ["linux", "unix", "unix/linux"],
    # data
    "postgresql": ["postgresql", "postgres", "psql"],
    "mysql": ["mysql"],
    "oracle": ["oracle", "oracle sql", "oracledb", "oracle db", "pl/sql", "plsql"],
    "mongodb": ["mongodb", "mongo", "mongo db"],
    "nosql": ["nosql", "no sql"],
    "cassandra": ["cassandra", "apache cassandra"],
    "elasticsearch": ["elasticsearch", "elastic search", "opensearch", "elk"],
    "redis": ["redis", "elasticache"],
    "spark": ["spark", "apache spark", "pyspark"],
    "snowflake": ["snowflake"],
    "airflow": ["airflow", "apache airflow"],
    "etl": ["etl", "elt", "data pipeline", "data pipelines"],
    # observability / quality
    "splunk": ["splunk"],
    "datadog": ["datadog", "data dog"],
    "grafana": ["grafana", "prometheus"],
    "junit": ["junit", "unit testing", "unit tests"],
    "mockito": ["mockito"],
    "testng": ["testng"],
    "observability": ["observability", "monitoring", "alerting", "telemetry"],
    # architecture / practice
    "system design": ["system design", "distributed systems", "scalable systems",
                      "high availability", "designing systems"],
    "caching": ["caching", "cache", "redis cache", "in-memory cache", "caching strategy"],
    "circuit breaker": ["circuit breaker", "resilience4j", "hystrix", "fault tolerance",
                        "resiliency", "resilience"],
    "api versioning": ["api versioning", "versioning strategy", "backward compatibility"],
    "oauth": ["oauth", "oauth2", "oauth 2.0", "openid connect", "oidc"],
    "jwt": ["jwt", "json web token", "token based authentication"],
    "security": ["security", "application security", "secure coding", "authentication",
                 "authorization", "authn", "authz"],
    "agile": ["agile", "scrum", "kanban", "sprint planning"],
    "tdd": ["tdd", "test driven development", "test-driven development"],
    "code review": ["code review", "code reviews", "peer review"],
    "mentoring": ["mentoring", "mentor", "coaching", "mentorship",
                  "technical leadership", "tech lead", "leading a team"],
    "on-call": ["on-call", "on call", "oncall", "incident response", "production support",
                "sre", "site reliability"],
    "event driven": ["event driven", "event-driven", "event driven architecture",
                     "event sourcing", "cqrs", "asynchronous processing"],
    "multithreading": ["multithreading", "multi-threading", "concurrency",
                       "concurrent programming", "threading", "parallel processing"],
    "design patterns": ["design patterns", "solid principles", "solid", "oop", "oops",
                        "object oriented", "object-oriented design", "ood"],
    "data structures": ["data structures", "algorithms", "dsa",
                        "data structures and algorithms"],
    "performance": ["performance tuning", "performance optimization", "latency optimization",
                    "throughput", "profiling", "jvm tuning"],
    # ml / ai
    "machine learning": ["machine learning", "ml", "predictive modeling"],
    "deep learning": ["deep learning", "neural networks", "lstm", "cnn"],
    "nlp": ["nlp", "natural language processing"],
    "llm": ["llm", "large language model", "genai", "generative ai", "rag", "prompt engineering"],
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow", "keras"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "pandas": ["pandas", "numpy"],
    # domain
    "fintech": ["fintech", "financial services", "wealth management", "banking",
                "payments", "capital markets", "trading", "brokerage", "investment"],
    "telecom": ["telecom", "telecommunications", "bss", "oss", "billing systems"],
}

# reverse index: surface form -> canonical
_SURFACE_TO_CANON: Dict[str, str] = {}
for _canon, _forms in ALIASES.items():
    _SURFACE_TO_CANON[_canon] = _canon
    for _f in _forms:
        _SURFACE_TO_CANON[_f] = _canon


# Skills that are "adjacent" — if you have the key, the value is a short,
# credible stretch rather than a fabrication. Used to rank learning suggestions.
ADJACENCY: Dict[str, List[str]] = {
    "kafka": ["rabbitmq", "activemq", "sqs", "kinesis", "event driven", "pubsub"],
    "aws": ["gcp", "azure"],
    "ecs": ["kubernetes", "docker", "eks"],
    "kubernetes": ["docker", "ecs", "helm", "terraform"],
    "spring boot": ["spring", "spring cloud", "hibernate", "microservices"],
    "oracle": ["postgresql", "mysql", "sql"],
    "mongodb": ["cassandra", "dynamodb", "nosql"],
    "redshift": ["snowflake", "spark", "etl"],
    "splunk": ["datadog", "grafana", "observability"],
    "java": ["kotlin", "scala"],
    "circuit breaker": ["resilience", "system design"],
    "machine learning": ["deep learning", "scikit-learn", "pandas"],
}


# --------------------------------------------------------------------------
# normalisation + matching
# --------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^a-z0-9+#./ -]+")
_WS_RE = re.compile(r"\s+")


def norm(text: str) -> str:
    s = (text or "").lower().strip()
    s = s.replace("&", " and ")
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def canonical(text: str) -> str:
    """Map a surface form to its canonical skill name (or the normalised text)."""
    n = norm(text)
    if n in _SURFACE_TO_CANON:
        return _SURFACE_TO_CANON[n]
    # try singular/plural and common suffix trimming
    for cand in (n.rstrip("s"), n + "s", n.replace("-", " "), n.replace(" ", "")):
        if cand in _SURFACE_TO_CANON:
            return _SURFACE_TO_CANON[cand]
    return n


def surface_forms(skill: str) -> List[str]:
    """All strings that would count as a mention of this skill."""
    c = canonical(skill)
    forms = set(ALIASES.get(c, []))
    forms.add(c)
    forms.add(norm(skill))
    return sorted(f for f in forms if f)


def _word_boundary_pattern(form: str) -> re.Pattern:
    esc = re.escape(form)
    # allow the separator characters in a multi-word form to vary (space/-/.)
    esc = esc.replace(r"\ ", r"[\s\-_/]+")
    lead = r"(?<![A-Za-z0-9+#])"
    trail = r"(?![A-Za-z0-9+#])"
    return re.compile(lead + esc + trail, re.I)


_PATTERN_CACHE: Dict[str, re.Pattern] = {}


def pattern_for(form: str) -> re.Pattern:
    if form not in _PATTERN_CACHE:
        _PATTERN_CACHE[form] = _word_boundary_pattern(form)
    return _PATTERN_CACHE[form]


def find_mentions(skill: str, text: str) -> List[re.Match]:
    """Every place `skill` (in any alias form) appears in `text`."""
    hits: List[re.Match] = []
    for form in surface_forms(skill):
        hits.extend(pattern_for(form).finditer(text))
    return hits


def mentioned_in(skill: str, text: str) -> bool:
    return bool(find_mentions(skill, text))


# --------------------------------------------------------------------------
# evidence quality
# --------------------------------------------------------------------------

_METRIC_RE = re.compile(
    r"(\d+(?:\.\d+)?\s*%|\b\d[\d,]*\s*(?:k|m|bn|mn|lakh|crore|million|billion)?\+?\b"
    r"|\bp\d{2}\b|\bqps\b|\brps\b|\btps\b|\bms\b|\bsla\b)",
    re.I,
)

_STRONG_VERBS = {
    "built", "designed", "architected", "led", "owned", "implemented", "migrated",
    "scaled", "reduced", "cut", "improved", "optimised", "optimized", "delivered",
    "launched", "drove", "automated", "rewrote", "refactored", "introduced",
    "established", "mentored", "orchestrated", "developed", "created", "shipped",
}

_WEAK_VERBS = {
    "worked", "helped", "assisted", "involved", "participated", "responsible",
    "handled", "supported", "used", "utilised", "utilized", "familiar", "exposure",
}


@dataclass
class Evidence:
    skill: str
    canonical: str
    level: str                      # strong | mentioned | listed_only | missing
    bullet_ids: List[str] = field(default_factory=list)
    in_skills_list: bool = False
    has_metric: bool = False
    sample: str = ""

    @property
    def is_gap(self) -> bool:
        return self.level in ("missing",)

    @property
    def is_weak(self) -> bool:
        return self.level in ("listed_only", "mentioned")


def assess(skill: str, bullets: Iterable, skills_list_text: str) -> Evidence:
    """Classify how well a resume already evidences one skill.

    `bullets` is any iterable of objects with `.bid` and a `.plain()` method.
    """
    canon = canonical(skill)
    ev = Evidence(skill=skill, canonical=canon, level="missing")
    ev.in_skills_list = mentioned_in(skill, skills_list_text)

    for b in bullets:
        text = b.plain()
        if not mentioned_in(skill, text):
            continue
        ev.bullet_ids.append(b.bid)
        if not ev.sample:
            ev.sample = text
        if _METRIC_RE.search(text):
            ev.has_metric = True

    if ev.bullet_ids:
        first_words = {w.lower().strip(",.") for w in (ev.sample.split()[:3] or [])}
        strong_verb = bool(first_words & _STRONG_VERBS)
        weak_verb = bool(first_words & _WEAK_VERBS)
        if (ev.has_metric or strong_verb) and not weak_verb:
            ev.level = "strong"
        else:
            ev.level = "mentioned"
    elif ev.in_skills_list:
        ev.level = "listed_only"
    else:
        ev.level = "missing"
    return ev


def adjacent_to(skill: str, owned: Set[str]) -> List[str]:
    """Which of the user's existing skills make `skill` a credible near-miss."""
    c = canonical(skill)
    related = set(ADJACENCY.get(c, []))
    for key, vals in ADJACENCY.items():
        if c in vals:
            related.add(key)
    return sorted(related & {canonical(o) for o in owned})


def extract_candidate_skills(text: str) -> List[str]:
    """Pull known skills out of free text (used as an LLM-free fallback)."""
    found: List[str] = []
    seen: Set[str] = set()
    for canon in ALIASES:
        if mentioned_in(canon, text) and canon not in seen:
            seen.add(canon)
            found.append(canon)
    return found


# --------------------------------------------------------------------------
# display names
# --------------------------------------------------------------------------

_DISPLAY_OVERRIDES = {
    "ci/cd": "CI/CD", "aws": "AWS", "gcp": "GCP", "sql": "SQL", "nosql": "NoSQL",
    "rest api": "REST APIs", "api gateway": "API Gateway", "jwt": "JWT",
    "oauth": "OAuth 2.0", "tdd": "TDD", "etl": "ETL", "llm": "LLMs",
    "nlp": "NLP", "ecs": "AWS ECS", "eks": "AWS EKS", "s3": "Amazon S3",
    "sqs": "Amazon SQS", "sns": "Amazon SNS", "ec2": "Amazon EC2",
    "c++": "C++", "node.js": "Node.js", "oops": "OOP", "on-call": "On-call",
    "system design": "System design / distributed systems",
    "performance": "Performance tuning", "security": "Application security",
    "observability": "Observability & monitoring", "kafka": "Apache Kafka",
    "mentoring": "Mentoring & technical leadership",
    "event driven": "Event-driven architecture",
    "design patterns": "OOP & design patterns",
    "data structures": "Data structures & algorithms",
    "circuit breaker": "Circuit breaking / resilience",
    "multithreading": "Concurrency & multithreading",
}


def display_name(skill: str) -> str:
    """Human-facing label for a skill (prefers the canonical concept name)."""
    c = canonical(skill)
    if c in _DISPLAY_OVERRIDES:
        return _DISPLAY_OVERRIDES[c]
    if c in ALIASES:
        return c.title() if c.islower() and " " in c else (
            c.capitalize() if c.islower() else c
        )
    return skill.strip()
