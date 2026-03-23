from modules.config import Config
from modules.database import Database
import logging

def truncate_db():
    logging.basicConfig(level=logging.INFO)
    try:
        config = Config("config.json")
        db = Database(config)
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('TRUNCATE TABLE events, sessions, faces CASCADE;')
            conn.commit()
        print('\nDatabase truncated successfully. You are ready for a clean run.')
    except Exception as e:
        print(f"\nError truncating database: {e}")

if __name__ == '__main__':
    truncate_db()
