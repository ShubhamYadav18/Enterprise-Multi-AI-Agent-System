"""
Search tools with pluggable provider architecture.

Uses a mock search provider by default. Can be replaced with Tavily
or any other provider by implementing the SearchProvider protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# Search Result Model
# ============================================================

class SearchResult(BaseModel):
    """A single search result."""

    title: str = Field(description="Result title")
    snippet: str = Field(description="Result snippet/summary")
    url: str = Field(description="Result URL")
    relevance: float = Field(default=0.0, description="Relevance score 0-1")


# ============================================================
# Search Provider Protocol
# ============================================================

class SearchProvider(ABC):
    """Abstract base class for search providers.

    Implement this interface to add new search backends
    (e.g., Tavily, Serper, Google) without changing agent code.
    """

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Execute a search query.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            List of SearchResult objects.
        """
        ...


# ============================================================
# Mock Search Provider
# ============================================================

class MockSearchProvider(SearchProvider):
    """Mock search provider with curated enterprise data.

    Returns realistic results from a local knowledge base of
    industry benchmarks, market data, and best practices.
    """

    def __init__(self) -> None:
        self._data: list[dict[str, Any]] = [
            {
                "title": "Industry Leave Policy Benchmarks 2025",
                "snippet": "The average US company provides 15-20 days of paid annual leave. "
                           "Tech companies average 20-25 days. Fortune 500 companies average 18 days. "
                           "European companies average 25-30 days. Sick leave averages 8-10 days in the US.",
                "url": "https://example.com/hr-benchmarks-2025",
                "keywords": ["leave", "vacation", "pto", "time off", "holiday", "sick"],
            },
            {
                "title": "Enterprise SaaS Market Report 2025",
                "snippet": "The global enterprise SaaS market is valued at $232B in 2025, growing at 13.7% CAGR. "
                           "Average enterprise deal size is $75K-$120K ARR. Customer retention rates average 90-95% "
                           "for top-tier providers. Net Revenue Retention (NRR) benchmark is 110-120%.",
                "url": "https://example.com/saas-market-2025",
                "keywords": ["saas", "market", "revenue", "enterprise", "deal", "pricing", "growth"],
            },
            {
                "title": "IT Security Best Practices 2025",
                "snippet": "Industry standard requires MFA adoption rates above 95%. Average breach cost is $4.45M. "
                           "SOC 2 compliance is required by 87% of enterprise buyers. Zero-trust architecture "
                           "adoption has reached 61% among Fortune 500 companies.",
                "url": "https://example.com/security-benchmarks-2025",
                "keywords": ["security", "compliance", "soc", "breach", "mfa", "zero trust"],
            },
            {
                "title": "Engineering Team Productivity Benchmarks",
                "snippet": "Average deployment frequency for high-performing teams is multiple times per day. "
                           "Lead time for changes averages 1-7 days. Change failure rate benchmark is below 15%. "
                           "Mean time to recovery (MTTR) should be under 1 hour for P1 incidents.",
                "url": "https://example.com/devops-benchmarks",
                "keywords": ["engineering", "devops", "deployment", "ci/cd", "productivity", "dora"],
            },
            {
                "title": "Employee Benefits Survey 2025",
                "snippet": "78% of companies offer remote work options. Average parental leave is 12 weeks (primary) "
                           "and 4 weeks (secondary). 65% of companies offer floating holidays. Health insurance "
                           "coverage averages $7,200/employee/year for employer contribution.",
                "url": "https://example.com/benefits-survey-2025",
                "keywords": ["benefits", "remote", "parental", "health", "insurance", "employee"],
            },
            {
                "title": "Sales Performance Metrics 2025",
                "snippet": "Average SaaS win rate is 25-35% for qualified deals. Sales cycle length: "
                           "SMB 14-30 days, Mid-Market 30-90 days, Enterprise 90-180 days. "
                           "Average commission rate is 8-12% of first-year revenue. "
                           "SDR to AE conversion rate averages 20-30%.",
                "url": "https://example.com/sales-benchmarks-2025",
                "keywords": ["sales", "commission", "win rate", "pipeline", "quota", "deal"],
            },
            {
                "title": "Expense Management Industry Standards",
                "snippet": "Average corporate travel spend is $1,200-$1,800 per trip. Hotel per-diem rates "
                           "average $150-$250 domestically. Meal per-diem: $60-$100/day. Companies using "
                           "automated expense management save 58% on processing costs.",
                "url": "https://example.com/expense-benchmarks",
                "keywords": ["expense", "travel", "per diem", "reimbursement", "corporate"],
            },
            {
                "title": "Workforce Utilization Analytics Report",
                "snippet": "Average employee utilization rate across industries is 60-70%. "
                           "Leave utilization rate (leave taken / leave entitled) averages 73%. "
                           "Effective working days per year: 230-240 (after leave and holidays). "
                           "Absenteeism rate benchmark: 2.5-3.5%.",
                "url": "https://example.com/utilization-report-2025",
                "keywords": ["utilization", "working days", "absenteeism", "productivity", "leave"],
            },
        ]

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search the mock data by keyword matching.

        Args:
            query: Search query string.
            max_results: Maximum results to return.

        Returns:
            Matching search results sorted by relevance.
        """
        query_lower = query.lower()
        scored_results: list[tuple[float, dict[str, Any]]] = []

        for entry in self._data:
            # Score based on keyword overlap
            keywords = entry["keywords"]
            score = sum(
                1.0 for kw in keywords
                if kw in query_lower
            )

            # Boost for title match
            if any(word in entry["title"].lower() for word in query_lower.split()):
                score += 0.5

            if score > 0:
                scored_results.append((score, entry))

        # Sort by score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)

        results = [
            SearchResult(
                title=entry["title"],
                snippet=entry["snippet"],
                url=entry["url"],
                relevance=min(score / 3.0, 1.0),  # Normalize to 0-1
            )
            for score, entry in scored_results[:max_results]
        ]

        logger.info(f"Mock search for '{query}' returned {len(results)} results")
        return results


# ============================================================
# Provider Singleton
# ============================================================

_provider: SearchProvider | None = None


def get_search_provider() -> SearchProvider:
    """Get the configured search provider.

    Returns MockSearchProvider by default. Override by calling
    set_search_provider() with a different implementation.
    """
    global _provider
    if _provider is None:
        _provider = MockSearchProvider()
    return _provider


def set_search_provider(provider: SearchProvider) -> None:
    """Set a custom search provider."""
    global _provider
    _provider = provider


# ============================================================
# LangChain Tool
# ============================================================

@tool
def search(query: str, max_results: int = 5) -> str:
    """Search for external information, industry benchmarks, and market data.

    Args:
        query: Search query string describing what to find.
        max_results: Maximum number of results to return (default 5).

    Returns:
        Formatted search results with titles, snippets, and URLs.
    """
    provider = get_search_provider()
    results = provider.search(query, max_results)

    if not results:
        return f"No results found for query: '{query}'"

    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(
            f"[{i}] {r.title}\n"
            f"    {r.snippet}\n"
            f"    Source: {r.url}\n"
            f"    Relevance: {r.relevance:.1%}"
        )

    return "\n\n".join(formatted)


SEARCH_TOOLS = [search]
