from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel, Field

from core.logging import get_logger

log = get_logger(__name__)
EMBEDDING_DIM = 1024


# Data Models

class MemoryEntry(BaseModel):

    content: str = Field(..., min_length=20)
    source: str = Field(default="agent")
    trust_level: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default="")


class MemorySearchResult(BaseModel):

    content: str
    source: str
    trust_level: float
    metadata: dict[str, Any]
    timestamp: str
    similarity_score: float = Field(description="Cosine similarity 0.0–1.0. Higher = more relevant.")


# Pinecone REST Client

class PineconeClient:

    # Pinecone's central API for index management and inference
    CONTROLLER_URL = "https://api.pinecone.io"

    def __init__(
        self,
        api_key: str,
        index_name: str,
        index_host: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._index_name = index_name
        self._index_host = index_host

        self._headers = {
            "Api-Key": api_key,
            "Content-Type": "application/json",
            "X-Pinecone-API-Version": "2024-07",
        }
        self._http = httpx.AsyncClient(
            headers=self._headers,
            timeout=30.0,  # Embedding + vector ops can take a few seconds
        )

    async def get_index_host(self) -> str:
        if self._index_host:
            return self._index_host

        response = await self._http.get(
            f"{self.CONTROLLER_URL}/indexes/{self._index_name}"
        )
        response.raise_for_status()
        data = response.json()
        self._index_host = data["host"]
        log.info("pinecone_index_host_resolved", host=self._index_host)
        return self._index_host

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self._http.post(
            f"{self.CONTROLLER_URL}/embed",
            json={
                "model": "multilingual-e5-large",
                "inputs": [{"text": t} for t in texts],
                "parameters": {
                    "input_type": "passage",  # "passage" for storing, "query" for searching
                    "truncate": "END",         # Truncate long texts from the end
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        return [item["values"] for item in data["data"]]

    async def embed_query(self, text: str) -> list[float]:
        response = await self._http.post(
            f"{self.CONTROLLER_URL}/embed",
            json={
                "model": "multilingual-e5-large",
                "inputs": [{"text": text}],
                "parameters": {
                    "input_type": "query",
                    "truncate": "END",
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["values"]

    async def upsert(self, vectors: list[dict]) -> int:
        host = await self.get_index_host()
        response = await self._http.post(
            f"{host}/vectors/upsert",
            json={"vectors": vectors},
        )
        response.raise_for_status()
        data = response.json()
        return data.get("upsertedCount", 0)

    async def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filter: dict | None = None,
        include_metadata: bool = True,
    ) -> list[dict]:
        host = await self.get_index_host()
        body: dict[str, Any] = {
            "vector": vector,
            "topK": top_k,
            "includeMetadata": include_metadata,
            "includeValues": False,  # Don't return the raw vectors (saves bandwidth)
        }
        if filter:
            body["filter"] = filter

        response = await self._http.post(f"{host}/query", json=body)
        response.raise_for_status()
        data = response.json()
        return data.get("matches", [])

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors by their IDs."""
        host = await self.get_index_host()
        await self._http.post(f"{host}/vectors/delete", json={"ids": ids})

    async def describe_index_stats(self) -> dict:
        """Return statistics about the index (total vector count, etc.)."""
        host = await self.get_index_host()
        response = await self._http.get(f"{host}/describe_index_stats")
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        """Close the HTTP client. Call at application shutdown."""
        await self._http.aclose()


# ─── Long-Term Memory ─────────────────────────────────────────────────────────

class LongTermMemory:

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
        index_host: str | None = None,
        min_content_length: int = 20,
    ) -> None:
        resolved_key = api_key or os.getenv("PINECONE_API_KEY")
        resolved_index = index_name or os.getenv("PINECONE_INDEX_NAME", "agent-memory")
        resolved_host = index_host or os.getenv("PINECONE_INDEX_HOST")

        if not resolved_key:
            raise ValueError(
                "Pinecone API key is required. "
                "Set PINECONE_API_KEY in your .env file. "
                "Get a key at: https://app.pinecone.io → API Keys"
            )

        self._client = PineconeClient(
            api_key=resolved_key,
            index_name=resolved_index,
            index_host=resolved_host,
        )
        self._min_content_length = min_content_length

        log.info(
            "long_term_memory_initialized",
            backend="pinecone_serverless",
            index=resolved_index,
            laptop_ram_used="~5MB (HTTP client only)",
        )

    async def store(self, entry: MemoryEntry) -> str | None:
        if len(entry.content) < self._min_content_length:
            log.debug(
                "memory_rejected_too_short",
                length=len(entry.content),
                minimum=self._min_content_length,
            )
            return None

        # Set timestamp if not already set
        if not entry.timestamp:
            entry.timestamp = datetime.utcnow().isoformat()

        # Deterministic ID from content hash — prevents duplicates
        vector_id = hashlib.sha256(entry.content.encode()).hexdigest()[:32]

        try:
            # Generate embedding via Pinecone Inference (server-side)
            embeddings = await self._client.embed([entry.content])
            vector = embeddings[0]

            metadata: dict[str, Any] = {
                "content": entry.content,       # Store content in metadata for retrieval
                "source": entry.source,
                "trust_level": entry.trust_level,
                "timestamp": entry.timestamp,
            }
            # Add extra metadata, converting complex values to strings
            for k, v in entry.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    metadata[k] = v
                else:
                    metadata[k] = str(v)

            count = await self._client.upsert([{
                "id": vector_id,
                "values": vector,
                "metadata": metadata,
            }])

            log.info(
                "long_term_memory_stored",
                vector_id=vector_id[:8] + "…",
                source=entry.source,
                trust_level=entry.trust_level,
                content_length=len(entry.content),
                upserted=count,
            )
            return vector_id

        except httpx.HTTPStatusError as e:
            log.error(
                "long_term_memory_store_failed",
                error=str(e),
                status_code=e.response.status_code,
            )
            return None

    async def retrieve(
        self,
        query: str,
        k: int = 5,
        min_trust: float = 0.0,
        min_similarity: float = 0.3,
        source_filter: str | None = None,
    ) -> list[MemorySearchResult]:
        try:
            # Embed the query using Pinecone Inference (query mode)
            query_vector = await self._client.embed_query(query)

            # Optional metadata filter (Pinecone filter syntax)
            pinecone_filter = None
            if source_filter:
                pinecone_filter = {"source": {"$eq": source_filter}}
            if min_trust > 0.0:
                trust_filter = {"trust_level": {"$gte": min_trust}}
                if pinecone_filter:
                    pinecone_filter = {"$and": [pinecone_filter, trust_filter]}
                else:
                    pinecone_filter = trust_filter

            # Query Pinecone — returns matches sorted by score descending
            matches = await self._client.query(
                vector=query_vector,
                top_k=k * 2,  # Fetch extra then post-filter by min_similarity
                filter=pinecone_filter,
                include_metadata=True,
            )

            results = []
            for match in matches:
                score = float(match.get("score", 0.0))
                if score < min_similarity:
                    continue

                meta = match.get("metadata", {})
                results.append(MemorySearchResult(
                    content=meta.get("content", ""),
                    source=meta.get("source", "unknown"),
                    trust_level=float(meta.get("trust_level", 1.0)),
                    metadata={
                        k: v for k, v in meta.items()
                        if k not in {"content", "source", "trust_level", "timestamp"}
                    },
                    timestamp=meta.get("timestamp", ""),
                    similarity_score=round(score, 4),
                ))

            # Already sorted by Pinecone, but cap at k
            top_results = results[:k]

            log.debug(
                "long_term_memory_retrieved",
                query=query[:60],
                results=len(top_results),
                top_score=top_results[0].similarity_score if top_results else 0,
            )
            return top_results

        except httpx.HTTPStatusError as e:
            log.error("long_term_memory_retrieve_failed", error=str(e))
            return []

    async def retrieve_as_context(
        self,
        query: str,
        k: int = 5,
        min_trust: float = 0.5,
    ) -> str:
        results = await self.retrieve(query, k=k, min_trust=min_trust)
        if not results:
            return ""

        lines = ["Relevant memories from past interactions:"]
        for i, r in enumerate(results, 1):
            trust_indicator = "✓" if r.trust_level >= 0.8 else "~"
            lines.append(
                f"  {i}. [{trust_indicator} {r.similarity_score:.0%} match] {r.content}"
            )

        return "\n".join(lines)

    async def store_batch(self, entries: list[MemoryEntry]) -> list[str | None]:
        valid = [e for e in entries if len(e.content) >= self._min_content_length]
        if not valid:
            return [None] * len(entries)

        try:
            for entry in valid:
                if not entry.timestamp:
                    entry.timestamp = datetime.utcnow().isoformat()

            batch_size = 96
            all_embeddings: list[list[float]] = []
            for i in range(0, len(valid), batch_size):
                batch = valid[i:i + batch_size]
                embeddings = await self._client.embed([e.content for e in batch])
                all_embeddings.extend(embeddings)

            vectors = []
            ids = []
            for entry, embedding in zip(valid, all_embeddings):
                vector_id = hashlib.sha256(entry.content.encode()).hexdigest()[:32]
                ids.append(vector_id)
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "content": entry.content,
                        "source": entry.source,
                        "trust_level": entry.trust_level,
                        "timestamp": entry.timestamp,
                        **{
                            k: v if isinstance(v, (str, int, float, bool)) else str(v)
                            for k, v in entry.metadata.items()
                        },
                    },
                })

            # Upsert in batches of 100 (Pinecone limit per request)
            total_upserted = 0
            for i in range(0, len(vectors), 100):
                count = await self._client.upsert(vectors[i:i + 100])
                total_upserted += count

            log.info(
                "long_term_memory_batch_stored",
                count=total_upserted,
                submitted=len(valid),
            )
            return ids

        except httpx.HTTPStatusError as e:
            log.error("long_term_memory_batch_failed", error=str(e))
            return [None] * len(entries)

    async def delete(self, vector_ids: list[str]) -> None:
        await self._client.delete(vector_ids)
        log.info("long_term_memory_deleted", count=len(vector_ids))

    async def get_stats(self) -> dict:
        return await self._client.describe_index_stats()

    async def close(self) -> None:
        await self._client.close()