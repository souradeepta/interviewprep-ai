import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells.append(nbf.v4.new_markdown_cell("# Knowledge Graphs\n\nObjectives: Graph representation, entity-relation modeling, multi-hop queries, reasoning, consistency"))

nb.cells.append(nbf.v4.new_code_cell("""from typing import Dict, List
from dataclasses import dataclass

# Level 1: Basic KG

class KnowledgeGraph:
    def __init__(self):
        self.entities = {}
        self.relations = []

    def add_entity(self, eid: str, name: str, etype: str):
        self.entities[eid] = {"name": name, "type": etype}

    def add_relation(self, from_id: str, rel: str, to_id: str):
        if from_id in self.entities and to_id in self.entities:
            self.relations.append((from_id, rel, to_id))

    def query_neighbors(self, eid: str, rel: str = None) -> List[Dict]:
        results = []
        for f, r, t in self.relations:
            if f == eid and (rel is None or r == rel):
                results.append({"id": t, "name": self.entities[t]["name"], "rel": r})
        return results

print('Level 1 - Basic KG:\\n')
kg = KnowledgeGraph()
kg.add_entity("alice", "Alice", "Person")
kg.add_entity("acme", "Acme", "Company")
kg.add_entity("boston", "Boston", "Location")
kg.add_relation("alice", "works_at", "acme")
kg.add_relation("acme", "located_in", "boston")

result = kg.query_neighbors("alice")
print(f"Alice's connections: {[n['name'] for n in result]}\\n"""))

nb.cells.append(nbf.v4.new_markdown_cell("**Key Points:** Entities are nodes. Relations are edges. Query neighbors. Simple but powerful."))

nb.cells.append(nbf.v4.new_code_cell("""# Level 2: Multi-hop Reasoning

class ReasoningKG(KnowledgeGraph):
    def multi_hop(self, start: str, path: List[str], max_depth: int = 3) -> List[str]:
        '''Follow relation chain.'''
        current = [start]
        for step, rel in enumerate(path):
            if step >= max_depth:
                break
            next_nodes = []
            for node in current:
                for n in self.query_neighbors(node, rel):
                    next_nodes.append(n["id"])
            current = next_nodes
        return current

    def infer_transitive(self):
        '''Infer: if A->B and B->C, then A->C (for transitive relations).'''
        inferred = []
        for (a, rel, b) in list(self.relations):
            if rel in ["located_in", "part_of"]:
                for (b2, rel2, c) in self.relations:
                    if b == b2 and rel2 == rel:
                        new = (a, rel, c)
                        if new not in self.relations and new not in inferred:
                            inferred.append(new)
        self.relations.extend(inferred)
        return len(inferred)

print('Level 2 - Reasoning:\\n')
kg = ReasoningKG()
kg.add_entity("p1", "Paris", "City")
kg.add_entity("france", "France", "Country")
kg.add_entity("europe", "Europe", "Continent")
kg.add_relation("p1", "located_in", "france")
kg.add_relation("france", "located_in", "europe")

inferred = kg.infer_transitive()
print(f"Inferred {inferred} new facts")

paris_continents = kg.multi_hop("p1", ["located_in", "located_in"])
print(f"Paris is transitively in: {[kg.entities[e]['name'] for e in paris_continents]}\\n"""))

nb.cells.append(nbf.v4.new_markdown_cell("**Key Takeaways:** Multi-hop queries traverse relations. Transitive reasoning infers new facts. Reduces redundancy."))

nb.cells.append(nbf.v4.new_code_cell("""# Example 1: Confidence-Scored Facts

@dataclass
class ScoredFact:
    from_ent: str
    rel: str
    to_ent: str
    confidence: float
    source: str

class ConfidentKG:
    def __init__(self):
        self.facts = []

    def add_fact(self, from_e, rel, to_e, conf, source):
        self.facts.append(ScoredFact(from_e, rel, to_e, conf, source))

    def query_confident(self, from_e, rel, min_conf=0.8):
        return [(f.to_ent, f.confidence) for f in self.facts 
                if f.from_ent == from_e and f.rel == rel and f.confidence >= min_conf]

print('Example 1 - Confidence Scoring:\\n')
kg = ConfidentKG()
kg.add_fact("treatment_a", "helps", "disease_x", 0.9, "clinical_trial")
kg.add_fact("treatment_a", "helps", "disease_y", 0.3, "anecdotal")

confident = kg.query_confident("treatment_a", "helps", min_conf=0.8)
print(f"High-confidence helps: {confident}\\n"""))

nb.cells.append(nbf.v4.new_markdown_cell("**Example 1 Key Points:** Score facts by confidence. Query above threshold. Uncertainty-aware reasoning."))

nb.cells.append(nbf.v4.new_code_cell("""# Example 2: Constraint Validation

class ConstrainedKG(KnowledgeGraph):
    def __init__(self):
        super().__init__()
        self.constraints = []

    def add_constraint(self, rule_name, check_fn):
        '''Add validation constraint.'''
        self.constraints.append((rule_name, check_fn))

    def validate(self):
        '''Check all constraints.'''
        violations = []
        for rule_name, check_fn in self.constraints:
            if not check_fn(self):
                violations.append(rule_name)
        return violations

print('\\nExample 2 - Constraints:\\n')
kg = ConstrainedKG()

# Add constraint: no person can be a location
def constraint_type_separation(kg):
    for from_id, rel, to_id in kg.relations:
        from_type = kg.entities[from_id]["type"]
        to_type = kg.entities[to_id]["type"]
        if from_type == "Person" and to_type == "Person" and rel == "located_in":
            return False
    return True

kg.add_constraint("person_cant_be_location", constraint_type_separation)
kg.add_entity("alice", "Alice", "Person")
kg.add_entity("boston", "Boston", "Location")
kg.add_relation("alice", "located_in", "boston")

violations = kg.validate()
print(f"Constraint violations: {violations if violations else 'None'}\\n"""))

nb.cells.append(nbf.v4.new_markdown_cell("**Example 2 Key Points:** Define constraints on valid relations. Validate before accepting. Prevent contradictions."))

nb.cells.append(nbf.v4.new_markdown_cell("""## Key Takeaways

**KG Components:**
- Entities: nodes in graph
- Relations: typed edges
- Properties: entity/edge attributes
- Rules: infer new facts

**Operations:**
- Query: direct neighbor lookup
- Multi-hop: follow relation chains
- Infer: apply rules to deduce facts
- Validate: check constraints

**Design Patterns:**
- Schema (entity types, relation types)
- Confidence scoring
- Constraint validation
- Transitive relation inference
- Caching for performance

**Related Concepts:** [[retrieval-augmented-generation]], [[semantic-memory]], [[agent-loops]]"""))

nbf.write(nb, '/home/sbisw/github/interviewprep-ml/agentic-ai/notebooks/knowledge-graphs.ipynb')
