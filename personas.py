"""
personas.py
-----------
Creative layer: turns dry cluster numbers into vivid customer personas with a
name, emoji, tagline, description, and a concrete marketing playbook.

Segments are matched to archetypes by their (income, spending) position
relative to the dataset medians, so this works for the real Kaggle data too.
"""

# Archetype templates keyed by (high_income, high_spending)
ARCHETYPES = {
    (True, True): {
        "persona": "Priya the Power Shopper",
        "emoji": "💎",
        "segment": "Premium / Target",
        "tagline": "High income, high spending — your VIPs.",
        "description": ("Affluent and happy to spend. Brand-loyal when treated "
                        "well. The single most valuable group to retain."),
        "playbook": ["Exclusive early access & launches",
                     "Tiered loyalty / VIP program",
                     "Premium bundles and concierge service",
                     "Personalised thank-you rewards"],
        "channel": "App push + premium email + events",
    },
    (True, False): {
        "persona": "Rohan the Reluctant Earner",
        "emoji": "🧐",
        "segment": "Careful (high income, low spend)",
        "tagline": "Money to spend, just not spending it here yet.",
        "description": ("High earning potential but low engagement. Convincing "
                        "them is the biggest untapped opportunity."),
        "playbook": ["Personalised product recommendations",
                     "Limited-time premium trials",
                     "Showcase quality & exclusivity, not discounts",
                     "Win trust with reviews & guarantees"],
        "channel": "Targeted email + retargeting ads",
    },
    (False, True): {
        "persona": "Aisha the Aspirational Spender",
        "emoji": "🛍️",
        "segment": "Careless (low income, high spend)",
        "tagline": "Loves to shop, watches the budget.",
        "description": ("Enthusiastic and engaged but income-limited. Great for "
                        "volume if you make spending easy and affordable."),
        "playbook": ["EMI / buy-now-pay-later options",
                     "Value bundles & combo offers",
                     "Flash sales and loyalty points",
                     "Gamified rewards to keep them coming"],
        "channel": "Social media + SMS offers",
    },
    (False, False): {
        "persona": "Sanjay the Sensible Saver",
        "emoji": "🪙",
        "segment": "Sensible / Frugal",
        "tagline": "Low income, low spending — careful with money.",
        "description": ("Price-sensitive and low-engagement. Don't overspend on "
                        "acquiring them; focus on light, cheap retention."),
        "playbook": ["Essentials and everyday-value messaging",
                     "Occasional deep-discount campaigns",
                     "Low-cost email newsletters only",
                     "Don't burn ad budget chasing them"],
        "channel": "Email newsletter + seasonal offers",
    },
}

STANDARD = {
    "persona": "Maya the Mainstream Middle",
    "emoji": "⚖️",
    "segment": "Standard / Average",
    "tagline": "Average income, average spending — the core majority.",
    "description": ("The biggest, steadiest group. Small nudges here move the "
                    "most total revenue simply because of volume."),
    "playbook": ["Steady seasonal promotions",
                 "Cross-sell and up-sell popular items",
                 "Membership / cashback to lift frequency",
                 "A/B test offers — small % gains scale big"],
    "channel": "Email + app + in-store signage",
}


def get_persona(avg_income, avg_spending, inc_med, spd_med,
                is_largest=False, near_center=False):
    """Pick a persona for a segment from its average income/spending."""
    if is_largest and near_center:
        return STANDARD
    return ARCHETYPES[(avg_income >= inc_med, avg_spending >= spd_med)]
