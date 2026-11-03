"""Extended SQLite Registry with SHA-256 verification, environment hashing,
atomic single-transaction promotion/rollback, single-ACTIVE constraint,
and baseline fallback for single-version rollback.
"""

import os
import sys
import sqlite3
import json
import difflib
import time
import hashlib
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple

from seae.task_analyzer import TaskSpec


def compute_code_sha256(source_code: str) -> str:
    """Compute normalized SHA-256 hash of candidate source code."""
    # Normalize CRLF to LF to ensure consistent cross-platform hashing
    norm = source_code.replace("\r\n", "\n").strip().encode("utf-8")
    return hashlib.sha256(norm).hexdigest()


def get_environment_versions() -> Dict[str, str]:
    """Capture runtime library versions affecting kernel binary behavior."""
    import numpy
    import cv2
    numba_ver = "unknown"
    try:
        import numba
        numba_ver = numba.__version__
    except Exception:
        pass

    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "numba": numba_ver,
    }


def make_environment_fingerprint(base_hw_fingerprint: str) -> str:
    """Combine hardware fingerprint and core compiler/math library versions."""
    vers = get_environment_versions()
    return f"{base_hw_fingerprint}__np{vers['numpy']}__nb{vers['numba']}__cv{vers['opencv']}"


@dataclass
class RegistryEntry:
    """Entry stored in the kernel registry."""
    entry_id: int
    kernel_name: str
    size_str: str
    dtype_str: str
    kernel_params_json: str
    hardware_fingerprint: str
    env_versions_json: str
    version: int
    candidate_id: str
    backend: str
    source_code: str
    code_sha256: str
    median_time_ms: float
    speedup_ratio: float
    peak_memory_mb: float
    verification_passed: bool
    status: str             # "ACTIVE", "ROLLED_BACK", "SUPERSEDED", "STAGED", "STALE"
    diff_from_prior: str
    created_at: float
    metadata_json: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KernelRegistry:
    """Registry maintaining optimal kernels with version history and rollback."""

    def __init__(self, db_path: str = "seae_registry.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create database tables and index if they do not exist."""
        with self._get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS kernel_cache (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                kernel_name TEXT NOT NULL,
                size_str TEXT NOT NULL,
                dtype_str TEXT NOT NULL,
                kernel_params_json TEXT NOT NULL,
                hardware_fingerprint TEXT NOT NULL,
                env_versions_json TEXT NOT NULL,
                version INTEGER NOT NULL,
                candidate_id TEXT NOT NULL,
                backend TEXT NOT NULL,
                source_code TEXT NOT NULL,
                code_sha256 TEXT NOT NULL,
                median_time_ms REAL NOT NULL,
                speedup_ratio REAL NOT NULL,
                peak_memory_mb REAL NOT NULL,
                verification_passed INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'ACTIVE',
                diff_from_prior TEXT,
                created_at REAL NOT NULL,
                metadata_json TEXT,
                UNIQUE(kernel_name, size_str, dtype_str, kernel_params_json, hardware_fingerprint, version)
            )
            """)
            conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_lookup
            ON kernel_cache(kernel_name, size_str, dtype_str, hardware_fingerprint, status)
            """)
            conn.commit()

    @staticmethod
    def _params_to_json(task_spec: TaskSpec) -> str:
        """Extract canonical JSON string of kernel parameters."""
        p = {}
        if task_spec.kernel_name == "gaussian_blur":
            p = {"ksize": task_spec.ksize, "sigma": task_spec.sigma}
        elif task_spec.kernel_name == "threshold":
            p = {"thresh": task_spec.thresh, "maxval": task_spec.maxval}
        return json.dumps(p, sort_keys=True)

    def compute_diff(self, old_code: str, new_code: str, old_label: str = "prior", new_label: str = "candidate") -> str:
        """Generate unified diff between two code strings."""
        old_lines = old_code.splitlines(keepends=True)
        new_lines = new_code.splitlines(keepends=True)
        diff = difflib.unified_diff(old_lines, new_lines, fromfile=old_label, tofile=new_label)
        return "".join(diff)

    def get_latest_active(
        self,
        kernel_name: str,
        size_str: str,
        hardware_fingerprint: str,
        dtype_str: str = "uint8",
        kernel_params: Optional[Dict[str, Any]] = None,
        check_sha: bool = True
    ) -> Optional[RegistryEntry]:
        """Fetch currently active optimal kernel, validating SHA-256 and environment."""
        params_json = json.dumps(kernel_params or {}, sort_keys=True)
        curr_env = get_environment_versions()

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT * FROM kernel_cache
                WHERE kernel_name = ? AND size_str = ? AND dtype_str = ?
                  AND kernel_params_json = ? AND hardware_fingerprint = ? AND status = 'ACTIVE'
                ORDER BY version DESC LIMIT 1
            """, (kernel_name, size_str, dtype_str, params_json, hardware_fingerprint))
            row = cur.fetchone()
            if row is None:
                return None

            entry = self._row_to_entry(row)

            # Check for SHA-256 integrity
            if check_sha:
                actual_sha = compute_code_sha256(entry.source_code)
                if actual_sha != entry.code_sha256:
                    # Cache corrupted or tampered! Mark stale/corrupt
                    cur.execute("UPDATE kernel_cache SET status = 'CORRUPTED' WHERE entry_id = ?", (entry.entry_id,))
                    conn.commit()
                    return None

            # Check environment version staleness (NumPy / Numba / OpenCV changes)
            cached_env = json.loads(entry.env_versions_json)
            if cached_env != curr_env:
                # Library versions have changed since kernel was compiled/optimized: mark STALE
                cur.execute("UPDATE kernel_cache SET status = 'STALE' WHERE entry_id = ?", (entry.entry_id,))
                conn.commit()
                return None

            return entry

    def get_history(
        self,
        kernel_name: str,
        size_str: str,
        hardware_fingerprint: str,
        dtype_str: str = "uint8",
        kernel_params: Optional[Dict[str, Any]] = None,
    ) -> List[RegistryEntry]:
        """Fetch all versions and audit trail for task."""
        params_json = json.dumps(kernel_params or {}, sort_keys=True)
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT * FROM kernel_cache
                WHERE kernel_name = ? AND size_str = ? AND dtype_str = ?
                  AND kernel_params_json = ? AND hardware_fingerprint = ?
                ORDER BY version ASC
            """, (kernel_name, size_str, dtype_str, params_json, hardware_fingerprint))
            rows = cur.fetchall()
            return [self._row_to_entry(r) for r in rows]

    def register_candidate(
        self,
        kernel_name: str,
        size_str: str,
        hardware_fingerprint: str,
        candidate_id: str,
        backend: str,
        source_code: str,
        median_time_ms: float,
        speedup_ratio: float,
        peak_memory_mb: float,
        verification_passed: bool,
        dtype_str: str = "uint8",
        kernel_params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_activate: bool = True
    ) -> RegistryEntry:
        """Register a new candidate in an atomic transaction.
        
        Guarantees:
        - At most one version is ACTIVE per (kernel_name, size_str, dtype_str, kernel_params, hw_fingerprint).
        - Prevents promotion of any code that was marked ROLLED_BACK.
        - Calculates and stores code SHA-256.
        """
        if not verification_passed:
            raise ValueError(f"Cannot register candidate {candidate_id}: verification failed.")

        params_json = json.dumps(kernel_params or {}, sort_keys=True)
        code_sha = compute_code_sha256(source_code)
        env_json = json.dumps(get_environment_versions())

        with self._get_connection() as conn:
            cur = conn.cursor()
            # Explicit transaction begins automatically in Python sqlite3 context manager

            # Rule: ROLLED_BACK versions must NEVER be auto-re-promoted
            cur.execute("""
                SELECT entry_id FROM kernel_cache
                WHERE kernel_name = ? AND size_str = ? AND dtype_str = ?
                  AND kernel_params_json = ? AND hardware_fingerprint = ?
                  AND code_sha256 = ? AND status = 'ROLLED_BACK'
            """, (kernel_name, size_str, dtype_str, params_json, hardware_fingerprint, code_sha))
            if cur.fetchone() is not None:
                raise ValueError(f"Candidate {candidate_id} matches a previously ROLLED_BACK implementation and cannot be re-promoted.")

            # Determine next version number
            cur.execute("""
                SELECT MAX(version) FROM kernel_cache
                WHERE kernel_name = ? AND size_str = ? AND dtype_str = ?
                  AND kernel_params_json = ? AND hardware_fingerprint = ?
            """, (kernel_name, size_str, dtype_str, params_json, hardware_fingerprint))
            max_v = cur.fetchone()[0]
            next_version = 1 if max_v is None else (max_v + 1)

            # Get current active to compute diff and mark superseded
            cur.execute("""
                SELECT * FROM kernel_cache
                WHERE kernel_name = ? AND size_str = ? AND dtype_str = ?
                  AND kernel_params_json = ? AND hardware_fingerprint = ? AND status = 'ACTIVE'
                ORDER BY version DESC LIMIT 1
            """, (kernel_name, size_str, dtype_str, params_json, hardware_fingerprint))
            prior_row = cur.fetchone()

            if prior_row is not None:
                diff_str = self.compute_diff(prior_row["source_code"], source_code, f"v{prior_row['version']}", f"v{next_version}")
                if auto_activate:
                    # In single transaction: supersede all currently active entries for this key
                    cur.execute("""
                        UPDATE kernel_cache SET status = 'SUPERSEDED'
                        WHERE kernel_name = ? AND size_str = ? AND dtype_str = ?
                          AND kernel_params_json = ? AND hardware_fingerprint = ? AND status = 'ACTIVE'
                    """, (kernel_name, size_str, dtype_str, params_json, hardware_fingerprint))
            else:
                diff_str = self.compute_diff("", source_code, "/dev/null", f"v{next_version}")

            status = "ACTIVE" if auto_activate else "STAGED"
            now = time.time()
            meta_str = json.dumps(metadata or {})

            cur.execute("""
                INSERT INTO kernel_cache (
                    kernel_name, size_str, dtype_str, kernel_params_json,
                    hardware_fingerprint, env_versions_json, version,
                    candidate_id, backend, source_code, code_sha256,
                    median_time_ms, speedup_ratio, peak_memory_mb,
                    verification_passed, status, diff_from_prior,
                    created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                kernel_name, size_str, dtype_str, params_json,
                hardware_fingerprint, env_json, next_version,
                candidate_id, backend, source_code, code_sha,
                median_time_ms, speedup_ratio, peak_memory_mb,
                1 if verification_passed else 0, status, diff_str,
                now, meta_str
            ))
            new_id = cur.lastrowid
            conn.commit()

            cur.execute("SELECT * FROM kernel_cache WHERE entry_id = ?", (new_id,))
            return self._row_to_entry(cur.fetchone())

    def rollback(
        self,
        kernel_name: str,
        size_str: str,
        hardware_fingerprint: str,
        dtype_str: str = "uint8",
        kernel_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[RegistryEntry], str]:
        """Roll back active kernel in an atomic transaction.
        
        Returns:
            (restored_entry_or_none, status_message)
            - If a prior version (v >= 1) exists: reactivates previous verified version.
            - If only a single version existed: marks it ROLLED_BACK and falls back to baseline ("FALLBACK_TO_BASELINE").
        """
        params_json = json.dumps(kernel_params or {}, sort_keys=True)

        with self._get_connection() as conn:
            cur = conn.cursor()

            # Find currently active entry
            cur.execute("""
                SELECT * FROM kernel_cache
                WHERE kernel_name = ? AND size_str = ? AND dtype_str = ?
                  AND kernel_params_json = ? AND hardware_fingerprint = ? AND status = 'ACTIVE'
                ORDER BY version DESC LIMIT 1
            """, (kernel_name, size_str, dtype_str, params_json, hardware_fingerprint))
            current = cur.fetchone()

            if current is None:
                return None, "NO_ACTIVE_VERSION"

            # Find previous valid version (verified passed and not rolled back)
            cur.execute("""
                SELECT * FROM kernel_cache
                WHERE kernel_name = ? AND size_str = ? AND dtype_str = ?
                  AND kernel_params_json = ? AND hardware_fingerprint = ?
                  AND version < ? AND verification_passed = 1 AND status != 'ROLLED_BACK'
                ORDER BY version DESC LIMIT 1
            """, (kernel_name, size_str, dtype_str, params_json, hardware_fingerprint, current["version"]))
            previous = cur.fetchone()

            if previous is None:
                # Rule 4: Single version rollback falls back to baseline!
                cur.execute("UPDATE kernel_cache SET status = 'ROLLED_BACK' WHERE entry_id = ?", (current["entry_id"],))
                conn.commit()
                return None, "FALLBACK_TO_BASELINE"

            # Reactivate previous version in single atomic transaction
            cur.execute("UPDATE kernel_cache SET status = 'ROLLED_BACK' WHERE entry_id = ?", (current["entry_id"],))
            cur.execute("UPDATE kernel_cache SET status = 'ACTIVE' WHERE entry_id = ?", (previous["entry_id"],))
            conn.commit()

            cur.execute("SELECT * FROM kernel_cache WHERE entry_id = ?", (previous["entry_id"],))
            return self._row_to_entry(cur.fetchone()), "RESTORED_PREVIOUS"

    def export_to_json(self, json_path: str) -> None:
        """Export registry snapshot to JSON file."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM kernel_cache ORDER BY entry_id ASC")
            rows = [dict(r) for r in cur.fetchall()]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)

    def import_from_json(self, json_path: str) -> int:
        """Import entries from a JSON file directly without re-verification (for testing/restoration)."""
        with open(json_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        imported = 0
        with self._get_connection() as conn:
            cur = conn.cursor()
            for e in entries:
                cur.execute("""
                    INSERT OR REPLACE INTO kernel_cache (
                        entry_id, kernel_name, size_str, dtype_str, kernel_params_json,
                        hardware_fingerprint, env_versions_json, version,
                        candidate_id, backend, source_code, code_sha256,
                        median_time_ms, speedup_ratio, peak_memory_mb,
                        verification_passed, status, diff_from_prior,
                        created_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    e.get("entry_id"), e["kernel_name"], e["size_str"], e.get("dtype_str", "uint8"),
                    e.get("kernel_params_json", "{}"), e["hardware_fingerprint"],
                    e.get("env_versions_json", "{}"), e["version"], e["candidate_id"],
                    e["backend"], e["source_code"], e.get("code_sha256", compute_code_sha256(e["source_code"])),
                    e["median_time_ms"], e["speedup_ratio"], e["peak_memory_mb"],
                    e["verification_passed"], e["status"], e.get("diff_from_prior"),
                    e["created_at"], e.get("metadata_json")
                ))
                imported += 1
            conn.commit()
        return imported

    def import_and_reverify_json(
        self,
        json_path: str,
        verifier: Any,
        benchmarker: Any,
        task_spec: TaskSpec,
    ) -> List[RegistryEntry]:
        """Import entries from JSON, strictly re-verifying and re-benchmarking before caching.
        
        Rule: JSON import must re-verify and re-benchmark, NEVER execute unverified code!
        """
        with open(json_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        promoted: List[RegistryEntry] = []
        for e in entries:
            # 1. Verify code SHA-256 matches claimed hash
            actual_sha = compute_code_sha256(e["source_code"])
            if actual_sha != e["code_sha256"]:
                continue

            # 2. Re-verify in isolated sandbox
            report = verifier.verify_candidate(
                e["candidate_id"],
                e["source_code"],
                task_spec,
                is_quantized=(e.get("dtype_str") in ("int8", "uint8"))
            )
            if not report.all_passed:
                continue

            # 3. Re-benchmark on this machine
            bench = benchmarker.benchmark_candidate(
                e["candidate_id"],
                e["source_code"],
                task_spec,
                is_pure_python=(e.get("backend") == "python_loops")
            )

            # 4. Safe promotion to local cache
            entry = self.register_candidate(
                kernel_name=e["kernel_name"],
                size_str=e["size_str"],
                hardware_fingerprint=e["hardware_fingerprint"],
                candidate_id=e["candidate_id"],
                backend=e["backend"],
                source_code=e["source_code"],
                median_time_ms=bench.median_time_ms,
                speedup_ratio=e.get("speedup_ratio", 1.0),
                peak_memory_mb=bench.peak_memory_mb,
                verification_passed=True,
                dtype_str=e.get("dtype_str", "uint8"),
                kernel_params=json.loads(e.get("kernel_params_json", "{}")),
                metadata={"imported_from_json": True, "re_benchmarked_ms": bench.median_time_ms}
            )
            promoted.append(entry)

        return promoted

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> RegistryEntry:
        return RegistryEntry(
            entry_id=row["entry_id"],
            kernel_name=row["kernel_name"],
            size_str=row["size_str"],
            dtype_str=row["dtype_str"],
            kernel_params_json=row["kernel_params_json"],
            hardware_fingerprint=row["hardware_fingerprint"],
            env_versions_json=row["env_versions_json"],
            version=row["version"],
            candidate_id=row["candidate_id"],
            backend=row["backend"],
            source_code=row["source_code"],
            code_sha256=row["code_sha256"],
            median_time_ms=row["median_time_ms"],
            speedup_ratio=row["speedup_ratio"],
            peak_memory_mb=row["peak_memory_mb"],
            verification_passed=bool(row["verification_passed"]),
            status=row["status"],
            diff_from_prior=row["diff_from_prior"] or "",
            created_at=row["created_at"],
            metadata_json=row["metadata_json"] or "{}",
        )
