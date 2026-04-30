from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.backend.config import (
    MONGODB_DATABASE,
    MONGODB_HISTORY_COLLECTION,
    MONGODB_URI,
)

try:
    import certifi
except ImportError:
    certifi = None

try:
    from pymongo import DESCENDING, MongoClient
    from bson import ObjectId
except ImportError:
    DESCENDING = None
    MongoClient = None
    ObjectId = None


_cached_collection: Any | None = None
_last_error = ""


def _collection() -> Any | None:
    global _cached_collection, _last_error

    if _cached_collection is not None:
        return _cached_collection

    if not MONGODB_URI or MongoClient is None:
        _last_error = "MONGODB_URI is missing." if not MONGODB_URI else "pymongo is not installed."
        return None

    try:
        client_kwargs = {"serverSelectionTimeoutMS": 10000}
        if certifi is not None:
            client_kwargs["tlsCAFile"] = certifi.where()

        client = MongoClient(MONGODB_URI, **client_kwargs)
        client.admin.command("ping")
        collection = client[MONGODB_DATABASE][MONGODB_HISTORY_COLLECTION]
        collection.create_index([("created_at", DESCENDING)])
        _last_error = ""
        _cached_collection = collection
        return _cached_collection
    except Exception as exc:
        message = str(exc)
        if "SSL handshake failed" in message:
            _last_error = (
                "MongoDB SSL handshake failed. Check Atlas Network Access/IP allowlist, "
                "connection string, and local TLS/certificate settings."
            )
        elif "Authentication failed" in message:
            _last_error = "MongoDB authentication failed. Check username and password."
        elif "ServerSelectionTimeoutError" in type(exc).__name__ or "Timeout" in message:
            _last_error = "MongoDB connection timed out. Check Atlas Network Access/IP allowlist."
        else:
            _last_error = message[:300]
        return None


def is_history_enabled() -> bool:
    return _collection() is not None


def history_status_message() -> str:
    _collection()
    return _last_error


def save_prediction(
    *,
    headline: str,
    content: str,
    label: str,
    confidence: float,
    probabilities: dict[str, float],
    branch_probabilities: dict[str, dict[str, float]],
) -> str | None:
    collection = _collection()
    if collection is None:
        return None

    document = {
        "headline": headline,
        "content": content,
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "branch_probabilities": branch_probabilities,
        "created_at": datetime.now(timezone.utc),
    }
    result = collection.insert_one(document)
    return str(result.inserted_id)


def list_predictions(limit: int = 20) -> list[dict[str, Any]]:
    collection = _collection()
    if collection is None:
        return []

    safe_limit = max(1, min(limit, 100))
    rows: list[dict[str, Any]] = []
    for document in collection.find().sort("created_at", DESCENDING).limit(safe_limit):
        created_at = document.get("created_at")
        rows.append(
            {
                "id": str(document.get("_id", "")),
                "headline": str(document.get("headline", "")),
                "content": str(document.get("content", "")),
                "label": str(document.get("label", "")),
                "confidence": float(document.get("confidence", 0.0)),
                "probabilities": document.get("probabilities", {}),
                "branch_probabilities": document.get("branch_probabilities", {}),
                "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else "",
            }
        )
    return rows


def delete_prediction(prediction_id: str) -> bool:
    collection = _collection()
    if collection is None or ObjectId is None:
        return False

    try:
        object_id = ObjectId(prediction_id)
    except Exception:
        return False

    result = collection.delete_one({"_id": object_id})
    return result.deleted_count == 1
