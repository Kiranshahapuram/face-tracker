from modules.config import Config
from modules.database import Database
import logging
import pandas as pd

def run_diagnostic_queries():
    try:
        config = Config("config.json")
        db = Database(config)
        
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                print("\n--- Faces Query (face_id, created_at) ---")
                cur.execute("SELECT id as face_id, first_seen as created_at FROM faces ORDER BY first_seen;")
                rows = cur.fetchall()
                print(f"{'face_id':<40} | {'created_at'}")
                print("-" * 70)
                for row in rows:
                    print(f"{str(row[0]):<40} | {row[1]}")
                print(f"\nTotal rows: {len(rows)}")
                
                print("\n--- Entry Counts per face_id ---")
                query = """
                SELECT e.face_id, COUNT(*) as entry_count, MIN(e.frame_number) as first_seen
                FROM events e
                WHERE e.event_type = 'entry'
                GROUP BY e.face_id
                ORDER BY first_seen;
                """
                cur.execute(query)
                rows = cur.fetchall()
                print(f"{'face_id':<40} | {'entries':<8} | {'first_seen'}")
                print("-" * 70)
                for row in rows:
                    print(f"{str(row[0]):<40} | {row[1]:<8} | {row[2]}")
            
    except Exception as e:
        print(f"\nError running queries: {e}")
            
    except Exception as e:
        print(f"\nError running queries: {e}")

if __name__ == '__main__':
    run_diagnostic_queries()
