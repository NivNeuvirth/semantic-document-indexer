import os
import logging
import psycopg2
from psycopg2.extras import execute_values
from typing import List
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.db_url = os.getenv("POSTGRES_URL")
        if not self.db_url:
            raise ValueError("❌ Missing Configuration: 'POSTGRES_URL' is not set in the .env file.")

    def get_connection(self):
        return psycopg2.connect(self.db_url)

    def setup_database(self):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                create_table_query = """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    chunk_text TEXT NOT NULL,
                    embedding double precision[],
                    filename TEXT NOT NULL,
                    split_strategy TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
                cur.execute(create_table_query)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_filename ON document_chunks(filename);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_doc_strategy ON document_chunks(split_strategy);")
                
            conn.commit()
            logger.info("✅ Database initialized (Custom Indexes Created).")
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ DB Setup failed: {e}")
            raise e
        finally:
            conn.close()

    def delete_existing_chunks(self, filename: str, strategy: str):
        """
        Duplicate Prevention: מחיקת נתונים ישנים לפני הכנסת חדשים.
        """
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                delete_query = """
                DELETE FROM document_chunks 
                WHERE filename = %s AND split_strategy = %s
                """
                cur.execute(delete_query, (filename, strategy))
                deleted_count = cur.rowcount
                
            conn.commit()
            if deleted_count > 0:
                logger.info(f"♻️  Cleaned up {deleted_count} old records for '{filename}'.")
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Cleanup failed: {e}")
            raise e
        finally:
            conn.close()

    def insert_chunks(self, filename: str, strategy: str, chunks: List[str], embeddings: List[List[float]]):
        if not chunks:
            return

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Delete existing chunks with same filename and strategy
                    delete_query = "DELETE FROM document_chunks WHERE filename = %s AND split_strategy = %s"
                    cur.execute(delete_query, (filename, strategy))
                    
                    insert_query = """
                    INSERT INTO document_chunks (filename, split_strategy, chunk_text, embedding)
                    VALUES %s
                    """
                    
                    data = [
                        (filename, strategy, chunk, embedding)
                        for chunk, embedding in zip(chunks, embeddings, strict=True)
                    ]
                    
                    execute_values(cur, insert_query, data)
                    
                conn.commit()
                logger.info(f"✅ Saved {len(data)} chunks to database.")
        except Exception as e:
            logger.exception(f"❌ Insert failed: {e}")
            raise

db_manager = DatabaseManager()