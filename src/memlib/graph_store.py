from abc import ABC, abstractmethod
from typing import Any
import json
import re

from neo4j import GraphDatabase

from memlib.types import GraphFact


class GraphStore(ABC):

    @abstractmethod
    def add_fact(self, fact: GraphFact) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_memory(self, memory_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        user_id: str,
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def clear_user(self, user_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class Neo4jGraphStore(GraphStore):

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        *,
        database: str | None = None,
    ) -> None:
        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
        )

        self.database = database

    def add_fact(self, fact: GraphFact) -> None:
        cypher = """
        MERGE (subject:Entity {
            user_id: $user_id,
            name: $subject
        })

        MERGE (object:Entity {
            user_id: $user_id,
            name: $object
        })

        MERGE (subject)-[relationship:RELATES {
            memory_id: $memory_id,
            relation: $relation,
            user_id: $user_id
        }]->(object)

        SET relationship.metadata = $metadata
        """

        parameters = {
            "user_id": fact.user_id,
            "subject": fact.subject,
            "relation": fact.relation,
            "object": fact.object,
            "memory_id": fact.memory_id,
            "metadata": json.dumps(fact.metadata or {}),
        }

        with self.driver.session(
            database=self.database
        ) as session:
            session.run(
                cypher,
                parameters,
            )

    def delete_memory(
        self,
        memory_id: str,
    ) -> None:
        cypher = """
        MATCH ()-[relationship:RELATES {
            memory_id: $memory_id
        }]->()

        DELETE relationship
        """

        parameters = {
            "memory_id": memory_id,
        }

        with self.driver.session(
            database=self.database
        ) as session:
            session.run(
                cypher,
                parameters,
            )

    def search(
        self,
        query: str,
        user_id: str,
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        search_terms = self._search_terms(query)

        if not search_terms:
            return []

        cypher = """
        MATCH (subject:Entity)-[relationship:RELATES]->(object:Entity)

        WHERE relationship.user_id = $user_id

        AND any(
            term IN $search_terms
            WHERE
                toLower(subject.name) CONTAINS term
                OR
                toLower(object.name) CONTAINS term
                OR
                toLower(relationship.relation) CONTAINS term
        )

        RETURN
            subject.name AS subject,
            relationship.relation AS relation,
            object.name AS object,
            relationship.memory_id AS memory_id,
            relationship.metadata AS metadata

        LIMIT $limit
        """

        parameters = {
            "user_id": user_id,
            "search_terms": search_terms,
            "limit": limit,
        }

        with self.driver.session(
            database=self.database
        ) as session:
            result = session.run(
                cypher,
                parameters,
            )

            return [
                {
                    "subject": record["subject"],
                    "relation": record["relation"],
                    "object": record["object"],
                    "memory_id": record["memory_id"],
                    "metadata": self._decode_metadata(
                        record["metadata"]
                    ),
                }
                for record in result
            ]

    def clear_user(
        self,
        user_id: str,
    ) -> None:
        cypher = """
        MATCH (entity:Entity {
            user_id: $user_id
        })

        DETACH DELETE entity
        """

        parameters = {
            "user_id": user_id,
        }

        with self.driver.session(
            database=self.database
        ) as session:
            session.run(
                cypher,
                parameters,
            )

    def close(self) -> None:
        """Close the Neo4j driver."""
        self.driver.close()

    @staticmethod
    def _decode_metadata(
        metadata: Any,
    ) -> dict[str, Any]:
        if metadata is None:
            return {}

        if isinstance(metadata, dict):
            return metadata

        if isinstance(metadata, str):
            try:
                value = json.loads(metadata)

                if isinstance(value, dict):
                    return value

            except json.JSONDecodeError:
                pass

        return {}

    @staticmethod
    def _search_terms(
        query: str,
    ) -> list[str]:
        words = re.findall(
            r"[a-zA-Z0-9_-]+",
            query.lower(),
        )

        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "do",
            "does",
            "for",
            "i",
            "in",
            "is",
            "it",
            "me",
            "my",
            "of",
            "on",
            "or",
            "the",
            "to",
            "what",
            "which",
            "who",
        }

        return [
            word
            for word in words
            if word not in stopwords
            and len(word) > 1
        ]