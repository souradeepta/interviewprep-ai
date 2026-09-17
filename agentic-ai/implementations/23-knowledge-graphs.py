"""Small in-memory knowledge graph with bounded multi-hop traversal."""

from collections import defaultdict, deque
from typing import DefaultDict, Dict, Iterable, List, Set, Tuple


Fact = Tuple[str, str, str]


class KnowledgeGraph:
    def __init__(self) -> None:
        self.entities: Set[str] = set()
        self.edges: DefaultDict[str, List[Tuple[str, str]]] = defaultdict(list)

    def add_entity(self, entity_id: str) -> None:
        if not entity_id:
            raise ValueError("entity_id must not be empty")
        self.entities.add(entity_id)

    def add_fact(self, subject: str, relation: str, object_id: str) -> None:
        if subject not in self.entities or object_id not in self.entities:
            raise KeyError("both entities must be registered before adding a fact")
        if not relation:
            raise ValueError("relation must not be empty")
        edge = (relation, object_id)
        if edge not in self.edges[subject]:
            self.edges[subject].append(edge)

    def neighbors(self, subject: str, relation: str = "") -> List[str]:
        return sorted(object_id for edge_relation, object_id in self.edges[subject] if not relation or edge_relation == relation)

    def reachable(self, start: str, relation: str, max_hops: int = 3) -> Set[str]:
        if start not in self.entities:
            raise KeyError(start)
        if max_hops < 0:
            raise ValueError("max_hops must be non-negative")
        queue = deque([(start, 0)])
        seen = {start}
        result: Set[str] = set()
        while queue:
            node, depth = queue.popleft()
            if depth == max_hops:
                continue
            for neighbor in self.neighbors(node, relation):
                result.add(neighbor)
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return result

    def facts(self) -> List[Fact]:
        return sorted((subject, relation, object_id) for subject, edges in self.edges.items() for relation, object_id in edges)

