#!/usr/bin/env python3
"""
Completion System Performance Monitor
"""

import sqlite3
from pathlib import Path


def monitor_performance():
    """Monitor completion system performance."""
    cache_db = Path(__file__).parent / "cache_data" / "completion_cache.db"

    if not cache_db.exists():
        print("Cache database not found")
        return

    try:
        with sqlite3.connect(str(cache_db)) as conn:
            # Get cache statistics
            cursor = conn.execute("SELECT COUNT(*) FROM completion_cache")
            total_entries = cursor.fetchone()[0]

            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM completion_cache
                WHERE datetime(timestamp) > datetime('now', '-1 hour')
            """
            )
            recent_entries = cursor.fetchone()[0]

            cursor = conn.execute(
                """
                SELECT environment, COUNT(*) FROM completion_cache
                GROUP BY environment ORDER BY COUNT(*) DESC LIMIT 5
            """
            )
            env_stats = cursor.fetchall()

            print("📊 Performance Statistics:")
            print(f"   Total cache entries: {total_entries}")
            print(f"   Recent entries (1h): {recent_entries}")
            print(f"   Cache utilization: {recent_entries / max(total_entries, 1):.1%}")
            print(
                f"   Top environments: {', '.join([f'{env}({count})' for env, count in env_stats])}"
            )

    except Exception as e:
        print(f"Error monitoring performance: {e}")


if __name__ == "__main__":
    monitor_performance()
