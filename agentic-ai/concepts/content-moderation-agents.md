# Content Moderation Agents

## Detailed Explanation

Moderation agents flag harmful content: toxicity, hate speech, misinformation, copyright, spam. Mechanisms: rule-based (regex), ML-based (classification), semantic (understanding meaning). Challenges: false positives (legitimate posts flagged), false negatives (miss violations), cultural context (same phrase offensive in one context, fine in another). Best for: social platforms, comment sections, user-generated content, platform safety.

## Best Practices

1. Threshold tuning (false positive vs negative)
2. Human review loop
3. Explainability
4. Context awareness
5. Appeal mechanism
6. Regional policies
7. Feedback loops
8. Regular retraining

## Code Examples

```python
class ContentModerator:
    def classify(self, content):
        toxicity = self._compute_toxicity(content)
        if toxicity > self.threshold:
            return {'action': 'flag', 'reason': 'toxic', 'confidence': toxicity}
        return {'action': 'allow'}
    
    def route(self, flagged_content):
        if flagged_content['confidence'] > 0.95:
            return {'action': 'auto_remove', 'appeal': True}
        else:
            return {'action': 'human_review'}
```

## Related Concepts

- Safety Alignment, Human Collaboration, Monitoring
