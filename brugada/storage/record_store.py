import json
import math
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = APP_ROOT / "data"
PAYLOAD_DIR = DATA_DIR / "records"
DB_PATH = DATA_DIR / "brugada_records.db"


def _now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        candidate = float(value)
    except Exception:  # noqa: BLE001
        return default
    if math.isnan(candidate) or math.isinf(candidate):
        return default
    return candidate


def _result_get(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "item"):
        try:
            return _to_jsonable(value.item())
        except Exception:  # noqa: BLE001
            pass

    if hasattr(value, "tolist"):
        try:
            return _to_jsonable(value.tolist())
        except Exception:  # noqa: BLE001
            pass

    return str(value)


def _normalize_payload(result: Any, record_uid: str, record_name: str, patient_id: str | None) -> dict:
    payload = _to_jsonable(result)
    if not isinstance(payload, dict):
        payload = {"value": payload}

    payload["record_uid"] = record_uid
    payload["record_name"] = record_name
    if patient_id:
        payload["patient_id"] = patient_id

    return payload


def _evidence_summary(clinician_explain: dict) -> str:
    evidence_counts = clinician_explain.get("evidence_counts", {}) if isinstance(clinician_explain, dict) else {}

    try:
        strong = int(float(evidence_counts.get("strong", 0)))
    except Exception:  # noqa: BLE001
        strong = 0

    try:
        moderate = int(float(evidence_counts.get("moderate", 0)))
    except Exception:  # noqa: BLE001
        moderate = 0

    try:
        weak = int(float(evidence_counts.get("weak", 0)))
    except Exception:  # noqa: BLE001
        weak = 0

    return f"S{strong}/M{moderate}/W{weak}"


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_records_feedback_columns(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(records)").fetchall()
    existing_columns = {str(row["name"]) for row in rows}

    migrations = [
        ("doctor_feedback", "ALTER TABLE records ADD COLUMN doctor_feedback TEXT"),
        ("doctor_feedback_note", "ALTER TABLE records ADD COLUMN doctor_feedback_note TEXT"),
        ("doctor_feedback_at", "ALTER TABLE records ADD COLUMN doctor_feedback_at TEXT"),
    ]

    for column_name, statement in migrations:
        if column_name not in existing_columns:
            conn.execute(statement)


def init_record_store(clear_existing: bool = False) -> None:
    """Initialize the record store database.

    Args:
        clear_existing: If True, clears all existing data for temporary storage mode.
    """
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_uid TEXT NOT NULL UNIQUE,
                record_name TEXT NOT NULL,
                patient_id TEXT,
                source_mode TEXT NOT NULL DEFAULT 'single',
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                label TEXT,
                risk TEXT,
                probability_raw REAL,
                probability_display REAL,
                decision_threshold_raw REAL,
                decision_threshold_display REAL,
                decision_stability_raw REAL,
                decision_stability_display REAL,
                gray_zone INTEGER NOT NULL DEFAULT 0,
                recommendation_tier TEXT,
                recommendation_text TEXT,
                evidence_summary TEXT,
                doctor_feedback TEXT,
                doctor_feedback_note TEXT,
                doctor_feedback_at TEXT,
                payload_path TEXT NOT NULL
            )
            """
        )
        _ensure_records_feedback_columns(conn)

        # Clear existing data if requested (temporary storage mode)
        if clear_existing:
            conn.execute("DELETE FROM records")
            conn.execute("DELETE FROM audit_log")
            # Also clear payload files
            import shutil
            if PAYLOAD_DIR.exists():
                shutil.rmtree(PAYLOAD_DIR)
            PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_uid TEXT,
                action TEXT NOT NULL,
                detail TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_records_created ON records(created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_records_status ON records(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_records_name ON records(record_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_records_feedback ON records(doctor_feedback)")
        conn.commit()


def _persist_payload(payload: dict, record_uid: str) -> str:
    payload_path = PAYLOAD_DIR / f"{record_uid}.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    return str(payload_path.relative_to(APP_ROOT)).replace("\\", "/")


def _write_audit(conn: sqlite3.Connection, record_uid: str | None, action: str, detail: str = "") -> None:
    conn.execute(
        "INSERT INTO audit_log (record_uid, action, detail, created_at) VALUES (?, ?, ?, ?)",
        (record_uid, action, detail, _now_utc_iso()),
    )


def save_record_result(
    record_name: str,
    result: Any,
    source_mode: str = "single",
    patient_id: str | None = None,
) -> str:
    init_record_store()
    record_uid = uuid.uuid4().hex

    clinician_explain = _result_get(result, "clinician_explain", {})
    if not isinstance(clinician_explain, dict):
        clinician_explain = {}

    label = str(_result_get(result, "label", "Unknown"))
    risk = str(_result_get(result, "risk", "Unknown"))

    probability_raw = _safe_float(_result_get(result, "probability", 0.0), 0.0)
    probability_display = _safe_float(_result_get(result, "display_probability", probability_raw), probability_raw)

    threshold_raw = _safe_float(_result_get(result, "decision_threshold", 0.05), 0.05)
    threshold_display = _safe_float(_result_get(result, "display_threshold", 0.35), 0.35)

    stability_raw = _safe_float(_result_get(result, "decision_stability", 0.0), 0.0)
    stability_display = _safe_float(_result_get(result, "display_decision_stability", stability_raw), stability_raw)

    gray_zone = 1 if bool(_result_get(result, "gray_zone", False)) else 0

    recommendation_tier = str(clinician_explain.get("recommendation_tier", "routine_clinical_correlation"))
    recommendation_text = str(clinician_explain.get("recommendation_text", "Clinical correlation is recommended."))
    evidence_summary = _evidence_summary(clinician_explain)

    payload = _normalize_payload(result, record_uid=record_uid, record_name=record_name, patient_id=patient_id)
    payload_rel_path = _persist_payload(payload, record_uid)

    now_iso = _now_utc_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO records (
                record_uid,
                record_name,
                patient_id,
                source_mode,
                status,
                created_at,
                updated_at,
                label,
                risk,
                probability_raw,
                probability_display,
                decision_threshold_raw,
                decision_threshold_display,
                decision_stability_raw,
                decision_stability_display,
                gray_zone,
                recommendation_tier,
                recommendation_text,
                evidence_summary,
                payload_path
            )
            VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_uid,
                record_name,
                patient_id,
                source_mode,
                now_iso,
                now_iso,
                label,
                risk,
                probability_raw,
                probability_display,
                threshold_raw,
                threshold_display,
                stability_raw,
                stability_display,
                gray_zone,
                recommendation_tier,
                recommendation_text,
                evidence_summary,
                payload_rel_path,
            ),
        )
        _write_audit(conn, record_uid, "created", f"source_mode={source_mode}")
        conn.commit()

    return record_uid


def save_batch_results(
    batch_results: list[dict],
    patient_id: str | None = None,
    patient_id_by_record: dict[str, str] | None = None,
) -> dict[str, str]:
    uid_by_record: dict[str, str] = {}
    patient_id_by_record = patient_id_by_record or {}

    for item in batch_results:
        if not isinstance(item, dict):
            continue

        record_name = str(item.get("record", "")).strip()
        if not record_name:
            continue

        payload = item.get("raw", item)
        if payload is None:
            continue

        try:
            record_patient_id = patient_id_by_record.get(record_name, patient_id)
            record_uid = save_record_result(
                record_name=record_name,
                result=payload,
                source_mode="batch",
                patient_id=record_patient_id,
            )
            uid_by_record[record_name] = record_uid
        except Exception:  # noqa: BLE001
            continue

    return uid_by_record


def get_record_counts() -> dict[str, int]:
    init_record_store()
    counts = {"active": 0, "archived": 0, "deleted": 0}

    with _connect() as conn:
        rows = conn.execute("SELECT status, COUNT(*) AS n FROM records GROUP BY status").fetchall()

    for row in rows:
        status = str(row["status"])
        if status in counts:
            counts[status] = int(row["n"])

    return counts


def list_records(status: str = "active", search: str = "", limit: int = 200) -> list[dict]:
    init_record_store()

    where = []
    args: list[Any] = []

    normalized_status = (status or "active").strip().lower()
    if normalized_status != "all":
        where.append("status = ?")
        args.append(normalized_status)

    search_text = (search or "").strip()
    if search_text:
        where.append("(record_name LIKE ? OR COALESCE(patient_id, '') LIKE ? OR COALESCE(label, '') LIKE ?)")
        wildcard = f"%{search_text}%"
        args.extend([wildcard, wildcard, wildcard])

    query = "SELECT * FROM records"
    if where:
        query += " WHERE " + " AND ".join(where)
    query += " ORDER BY created_at DESC LIMIT ?"
    args.append(int(limit))

    with _connect() as conn:
        rows = conn.execute(query, args).fetchall()

    items = []
    for row in rows:
        items.append(
            {
                "record_uid": row["record_uid"],
                "record_name": row["record_name"],
                "patient_id": row["patient_id"] or "",
                "source_mode": row["source_mode"],
                "status": row["status"],
                "created_at": row["created_at"],
                "label": row["label"] or "Unknown",
                "risk": row["risk"] or "Unknown",
                "probability_raw": _safe_float(row["probability_raw"], 0.0),
                "probability_display": _safe_float(row["probability_display"], 0.0),
                "decision_stability_raw": _safe_float(row["decision_stability_raw"], 0.0),
                "decision_stability_display": _safe_float(row["decision_stability_display"], 0.0),
                "gray_zone": bool(int(row["gray_zone"] or 0)),
                "recommendation_tier": row["recommendation_tier"] or "routine_clinical_correlation",
                "recommendation_text": row["recommendation_text"] or "Clinical correlation is recommended.",
                "evidence_summary": row["evidence_summary"] or "S0/M0/W0",
                "doctor_feedback": (row["doctor_feedback"] or ""),
                "doctor_feedback_note": (row["doctor_feedback_note"] or ""),
                "doctor_feedback_at": (row["doctor_feedback_at"] or ""),
            }
        )

    return items


def get_record_payload(record_uid: str) -> dict | None:
    init_record_store()

    with _connect() as conn:
        row = conn.execute(
            "SELECT payload_path FROM records WHERE record_uid = ? LIMIT 1",
            (record_uid,),
        ).fetchone()

    if row is None:
        return None

    payload_rel = str(row["payload_path"])
    payload_path = APP_ROOT / payload_rel
    if not payload_path.exists():
        return None

    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None

    return payload if isinstance(payload, dict) else None


def update_record_status(record_uid: str, new_status: str) -> bool:
    init_record_store()

    allowed = {"active", "archived", "deleted"}
    status = (new_status or "").strip().lower()
    if status not in allowed:
        raise ValueError(f"Unsupported status: {new_status}")

    with _connect() as conn:
        cur = conn.execute(
            "UPDATE records SET status = ?, updated_at = ? WHERE record_uid = ?",
            (status, _now_utc_iso(), record_uid),
        )
        changed = cur.rowcount > 0
        if changed:
            _write_audit(conn, record_uid, "status_changed", f"new_status={status}")
        conn.commit()

    return changed


def update_record_status_bulk(record_uids: list[str], new_status: str) -> int:
    init_record_store()

    allowed = {"active", "archived", "deleted"}
    status = (new_status or "").strip().lower()
    if status not in allowed:
        raise ValueError(f"Unsupported status: {new_status}")

    cleaned_uids = [str(uid).strip() for uid in (record_uids or []) if str(uid).strip()]
    if not cleaned_uids:
        return 0

    changed_count = 0
    now_iso = _now_utc_iso()
    with _connect() as conn:
        for record_uid in cleaned_uids:
            cur = conn.execute(
                "UPDATE records SET status = ?, updated_at = ? WHERE record_uid = ?",
                (status, now_iso, record_uid),
            )
            if cur.rowcount > 0:
                changed_count += 1
                _write_audit(conn, record_uid, "status_changed", f"new_status={status}")
        conn.commit()

    return changed_count


def update_record_patient_id(record_uid: str, patient_id: str | None) -> bool:
    init_record_store()

    normalized_patient_id = (patient_id or "").strip()
    patient_value = normalized_patient_id if normalized_patient_id else None

    with _connect() as conn:
        cur = conn.execute(
            "UPDATE records SET patient_id = ?, updated_at = ? WHERE record_uid = ?",
            (patient_value, _now_utc_iso(), record_uid),
        )
        changed = cur.rowcount > 0
        if changed:
            detail = f"patient_id={normalized_patient_id}" if normalized_patient_id else "patient_id_cleared"
            _write_audit(conn, record_uid, "patient_id_updated", detail)
        conn.commit()

    return changed


def update_record_feedback(record_uid: str, feedback: str | None, note: str | None = None) -> bool:
    init_record_store()

    feedback_value = (feedback or "").strip().lower()
    if feedback_value == "":
        feedback_value = None

    if feedback_value not in {"agree", "disagree", None}:
        raise ValueError("Feedback must be one of: agree, disagree, or empty")

    note_value = (note or "").strip()
    if not note_value:
        note_value = None

    feedback_at = _now_utc_iso() if feedback_value else None
    now_iso = _now_utc_iso()

    with _connect() as conn:
        cur = conn.execute(
            """
            UPDATE records
            SET doctor_feedback = ?, doctor_feedback_note = ?, doctor_feedback_at = ?, updated_at = ?
            WHERE record_uid = ?
            """,
            (feedback_value, note_value, feedback_at, now_iso, record_uid),
        )
        changed = cur.rowcount > 0
        if changed:
            row = conn.execute(
                "SELECT payload_path FROM records WHERE record_uid = ? LIMIT 1",
                (record_uid,),
            ).fetchone()
            if row is not None:
                payload_path = APP_ROOT / str(row["payload_path"])
                if payload_path.exists():
                    try:
                        payload = json.loads(payload_path.read_text(encoding="utf-8"))
                    except Exception:  # noqa: BLE001
                        payload = {}
                    if not isinstance(payload, dict):
                        payload = {}

                    payload["doctor_feedback"] = {
                        "label": feedback_value,
                        "note": note_value,
                        "updated_at": feedback_at,
                    }
                    payload_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

            if feedback_value:
                detail = f"feedback={feedback_value}"
                if note_value:
                    detail += ";note_present=1"
                _write_audit(conn, record_uid, "doctor_feedback_updated", detail)
            else:
                _write_audit(conn, record_uid, "doctor_feedback_cleared", "")
        conn.commit()

    return changed
