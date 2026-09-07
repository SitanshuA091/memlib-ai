"""Knowledge graph storage interfaces and Neo4j implementation."""

from abc import ABC, abstractmethod
from typing import Any

from neo4j import GraphDatabase

from memlib.types import GraphFact


class GraphStore(ABC):

    @abstractmethod
    def add_fact(self, fact: GraphFact) -> None:
        """Add or update a graph relationship."""
        raise NotImplementedError

    @abstractmethod
    def delete_memory(self, memory_id: str) -> None:
        """
        Delete graph relationships derived from a canonical memory.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        user_id: str,
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieve graph relationships relevant to a query."""
        raise NotImplementedError

    @abstractmethod
    def clear_user(self, user_id: str) -> None:
        """Remove graph data belonging to a user."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close backend resources."""
        raise NotImplementedError


class Neo4jGraphStore(GraphStore):

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        *,
        database: str = "neo4j",
    ) -> None:
        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
        )

        self.database = database

    def add_fact(self, fact: GraphFact) -> None:

        query = """
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
            "metadata": fact.metadata,
        }

        with self.driver.session(
            database=self.database
        ) as session:
            session.run(
                query,
                parameters,
            )

    def delete_memory(self, memory_id: str) -> None:
        query = """
        MATCH ()-[relationship:RELATES {
            memory_id: $memory_id
        }]->()

        DELETE relationship
        """

        with self.driver.session(
            database=self.database
        ) as session:
            session.run(
                query,
                memory_id=memory_id,
            )

    def search(
        self,
        query: str,
        user_id: str,
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Retrieve graph relationships whose entities or relationship
        names contain terms from the query.

        This is intentionally a simple V1 graph search implementation.
        More advanced entity extraction / Cypher generation can be added
        later without changing the GraphStore interface.
        """

        cypher = """
        MATCH (subject:Entity)-[relationship:RELATES]->(object:Entity)

        WHERE relationship.user_id = $user_id

        AND (
            toLower(subject.name) CONTAINS toLower($query)
            OR
            toLower(object.name) CONTAINS toLower($query)
            OR
            toLower(relationship.relation) CONTAINS toLower($query)
        )

        RETURN
            subject.name AS subject,
            relationship.relation AS relation,
            object.name AS object,
            relationship.memory_id AS memory_id,
            relationship.metadata AS metadata

        LIMIT $limit
        """

        with self.driver.session(
            database=self.database
        ) as session:
            result = session.run(
                cypher,
                user_id=user_id,
                query=query,
                limit=limit,
            )

            return [
                {
                    "subject": record["subject"],
                    "relation": record["relation"],
                    "object": record["object"],
                    "memory_id": record["memory_id"],
                    "metadata": (
                        record["metadata"] or {}
                    ),
                }
                for record in result
            ]

    def clear_user(self, user_id: str) -> None:

        query = """
        MATCH (entity:Entity {
            user_id: $user_id
        })

        DETACH DELETE entity
        """

        with self.driver.session(
            database=self.database
        ) as session:
            session.run(
                query,
                user_id=user_id,
            )

    def close(self) -> None:
        """Close the Neo4j driver."""
        self.driver.close()