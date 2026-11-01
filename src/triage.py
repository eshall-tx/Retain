"""
Customer Message Triage Module for Retain.
Evaluates customer support messages for urgency, refund risk, and human escalation.
Uses TypeSafe AI's Jev model if JEV_API_KEY is present; otherwise falls back to VADER + rules.
"""

import os
import re
import requests

# Try importing free VADER sentiment analyzer
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_ANALYZER = SentimentIntensityAnalyzer()
except ImportError:
    _VADER_ANALYZER = None


def triage(text: str) -> dict:
    """
    Evaluates an incoming customer message to determine:
    1. Urgency: 'low', 'medium', or 'high'
    2. Refund Risk: 'low', 'medium', or 'high'
    3. Needs a Human: 'yes' or 'no'
    
    If the environment variable JEV_API_KEY is set, calls TypeSafe AI's Jev API.
    Otherwise (or on any network/API error), falls back to local VADER sentiment
    and simple keyword rules.
    """
    clean_text = (text or "").strip()
    if not clean_text:
        return {
            "urgency": "low",
            "refund_risk": "low",
            "needs_human": "no",
            "engine": "Local Rules (Empty Message)",
            "details": "No text provided"
        }

    # =========================================================================
    # OPTIONAL: TYPESAFE AI JEV INTEGRATION (Can be removed without breaking app)
    # =========================================================================
    jev_api_key = os.environ.get("JEV_API_KEY")
    if jev_api_key:
        try:
            endpoint = "https://api.typesafe.ai/v1/systemone"
            headers = {
                "Authorization": f"Bearer {jev_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "jev-latest",
                "state": clean_text,
                "questions": {
                    "urgency": {
                        "type": "choice",
                        "instructions": "Evaluate the urgency level of this customer message.",
                        "criteria": {
                            "low": "Casual question, feedback, standard inquiry, polite tone",
                            "medium": "Slow connection, unresolved bug, billing question, minor frustration",
                            "high": "Threatening cancellation, immediate help needed, outrage, service outage"
                        }
                    },
                    "refund_risk": {
                        "type": "choice",
                        "instructions": "Determine the refund or charge dispute risk.",
                        "criteria": {
                            "low": "No mention of money back, refund, or billing dispute",
                            "medium": "Questions about unusual charges or seeking small bill correction",
                            "high": "Explicitly demanding refund, chargeback, or compensation"
                        }
                    },
                    "needs_human": {
                        "type": "choice",
                        "instructions": "Does this message need human agent escalation?",
                        "criteria": {
                            "yes": "High churn risk, angry tone, complex negotiation, cancellation request",
                            "no": "Routine question easily answered by automated FAQ or documentation"
                        }
                    }
                }
            }

            response = requests.post(endpoint, headers=headers, json=payload, timeout=5)
            if response.status_code == 200:
                data = response.json()
                answers = data.get("answers", {})

                urgency = answers.get("urgency", {}).get("choice", "medium").lower()
                refund_risk = answers.get("refund_risk", {}).get("choice", "low").lower()
                needs_human = answers.get("needs_human", {}).get("choice", "no").lower()

                return {
                    "urgency": urgency,
                    "refund_risk": refund_risk,
                    "needs_human": needs_human,
                    "engine": "TypeSafe AI (Jev API)",
                    "details": f"Model: {data.get('model', 'jev-latest')}"
                }
        except Exception:
            # Silently fall back to local rules on any connection/parsing error
            pass
    # =========================================================================
    # END OF JEV INTEGRATION
    # =========================================================================

    # =========================================================================
    # LOCAL FALLBACK: VADER SENTIMENT + SIMPLE KEYWORD RULES
    # =========================================================================
    lower_text = clean_text.lower()

    # 1. Compute VADER sentiment if available
    compound_score = 0.0
    if _VADER_ANALYZER is not None:
        vader_scores = _VADER_ANALYZER.polarity_scores(clean_text)
        compound_score = vader_scores.get("compound", 0.0)

    # 2. Keyword patterns
    high_urgency_words = [
        "cancel", "leaving", "immediately", "asap", "urgent", "emergency",
        "worst", "terrible", "horrible", "lawyer", "attorney", "sue",
        "unacceptable", "furious", "broken", "outage", "switched"
    ]
    medium_urgency_words = [
        "slow", "issue", "problem", "delay", "help", "error",
        "dispute", "question", "waiting", "fix", "broken"
    ]

    high_refund_words = [
        "refund", "money back", "overcharged", "unauthorized", "chargeback",
        "compensation", "scam", "rip off", "reimburse", "stolen"
    ]
    medium_refund_words = [
        "dispute", "wrong charge", "incorrect bill", "hidden fee", "rate increase",
        "price hike", "expensive", "bill error", "extra charge", "unexpected charge"
    ]

    human_words = [
        "human", "representative", "agent", "manager", "person", "speak to",
        "talk to", "escalate", "supervisor", "operator"
    ]

    # Evaluate Urgency
    if any(re.search(rf"\b{re.escape(w)}\b", lower_text) for w in high_urgency_words) or compound_score <= -0.55:
        urgency = "high"
    elif any(re.search(rf"\b{re.escape(w)}\b", lower_text) for w in medium_urgency_words) or compound_score < -0.1:
        urgency = "medium"
    else:
        urgency = "low"

    # Evaluate Refund Risk
    if any(re.search(rf"\b{re.escape(w)}\b", lower_text) for w in high_refund_words):
        refund_risk = "high"
    elif any(re.search(rf"\b{re.escape(w)}\b", lower_text) for w in medium_refund_words):
        refund_risk = "medium"
    else:
        refund_risk = "low"

    # Evaluate Needs a Human
    demands_human = any(re.search(rf"\b{re.escape(w)}\b", lower_text) for w in human_words)
    if demands_human or urgency == "high" or refund_risk == "high":
        needs_human = "yes"
    elif urgency == "medium" and compound_score < -0.3:
        needs_human = "yes"
    else:
        needs_human = "no"

    sentiment_desc = "Neutral/Positive"
    if compound_score <= -0.5:
        sentiment_desc = f"Very Negative ({compound_score:.2f})"
    elif compound_score < 0:
        sentiment_desc = f"Negative ({compound_score:.2f})"

    return {
        "urgency": urgency,
        "refund_risk": refund_risk,
        "needs_human": needs_human,
        "engine": "Local VADER + Keyword Rules (Fallback)",
        "details": f"Sentiment: {sentiment_desc}"
    }
