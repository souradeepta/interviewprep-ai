"""
Auto-generated from 48-recommendation-agents.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # Recommendation Agents
# Objectives: Core patterns, implementation, optimization
# ======================================================================

class Recommender:
    def score_item(self, user_profile, item):
        score = 0
        if item['category'] in user_profile['liked']:
            score += 0.5
        if item['rating'] > 4:
            score += 0.3
        return score

rec = Recommender()
profile = {'liked': ['action']}
item = {'category': 'action', 'rating': 4.5}
print(f'Score: {rec.score_item(profile, item)}')


class PersonalizedRanker:
    def rank(self, user_id, items):
        recommender = Recommender()
        profile = {'liked': ['action', 'sci-fi']}
        scored = [(item, recommender.score_item(profile, item)) for item in items]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [item for item, score in scored]

ranker = PersonalizedRanker()
items = [{'category': 'action', 'rating': 4.5}, {'category': 'drama', 'rating': 3.0}]
ranked = ranker.rank('user1', items)
print(f'Top recommendation: {ranked[0]}')


class DiverseRanker:
    def add_diversity(self, ranked_items):
        # Add diverse items to top-10
        selected = ranked_items[:5]
        for item in ranked_items[5:]:
            if item['category'] not in [s['category'] for s in selected]:
                selected.append(item)
                if len(selected) >= 10:
                    break
        return selected

ranker = DiverseRanker()
print('Diversity added to recommendations')

