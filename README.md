# Semantic Document Indexer

A robust Python-based agentic tool designed to ingest, chunk, and semantically index documents for Retrieval-Augmented Generation (RAG) applications. This project leverages Google's Gemini models for high-quality embeddings and PostgreSQL for efficient vector storage.

## Project Description

The Semantic Document Indexer automates the process of preparing unstructured text data for AI applications. It handles the entire pipeline:
1.  **Ingestion**: Loads documents from various formats (PDF, DOCX, etc.).
2.  **Cleaning**: Preprocesses text to remove noise and standardize formatting.
3.  **Chunking**: Splits text into manageable segments using configurable strategies (Fixed-size, Sentence-based, Paragraph-based).
4.  **Embedding**: Generates vector embeddings for each chunk using Google's Gemini `text-embedding-004` model.
5.  **Indexing**: Stores proper chunks and their corresponding embeddings into a PostgreSQL database for retrieval.

## Prerequisites

Before setting up the project, ensure you have the following:

*   **Python 3.10+** (Recommended)
*   **PostgreSQL** installed.
*   **API Keys**:
    *   `GEMINI_API_KEY`: For generating embeddings via Google GenAI.
    *   `POSTGRES_URL`: Connection string for your PostgreSQL database.

### Local PostgreSQL Setup

If you don't have PostgreSQL installed, follow these steps:

1.  **Download and Install**:
    -   **Windows/macOS**: Download the installer from [postgresql.org](https://www.postgresql.org/download/).
    -   **Linux (Debian/Ubuntu)**:
        ```bash
        sudo apt update
        sudo apt install postgresql postgresql-contrib
        ```

2.  **Create Database and User**:
    Open your terminal or command prompt (on Windows, use SQL Shell or pgAdmin) and run:
    ```sql
    -- Create a new database
    CREATE DATABASE your_db_name;

    -- Create a new user with a password
    CREATE USER your_user WITH PASSWORD 'your_password';

    -- Grant privileges
    GRANT ALL PRIVILEGES ON DATABASE your_db_name TO your_user;
    -- For PostgreSQL 15+, you may also need to grant schema usage
    GRANT ALL ON SCHEMA public TO your_user;
    ```
    *Update your `.env` file with these credentials.*

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/NivNeuvirth/semantic-document-indexer.git
    cd semantic-document-indexer
    ```

2.  **Set up a Virtual Environment**:
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**:
    Create a `.env` file in the root directory:
    ```ini
    # .env
    GEMINI_API_KEY=your_google_api_key
    POSTGRES_URL=postgresql://user:password@localhost:5432/your_db_name
    ```

## Database Schema

The project uses a single table `document_chunks` to store the indexed data. The schema is automatically created if it doesn't exist.

### Table: `document_chunks`

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `SERIAL PRIMARY KEY` | Unique identifier for each chunk. |
| `chunk_text` | `TEXT` | The actual text content of the chunk. |
| `embedding` | `double precision[]` | The vector embedding of the chunk (768 dimensions, corresponding to the Gemini text-embedding-004 model output). |
| `filename` | `TEXT` | Name of the source file. |
| `split_strategy`| `TEXT` | The strategy used to split the text (e.g., 'fixed', 'paragraph'). |
| `created_at` | `TIMESTAMP` | Timestamp of insertion (Default: Current Time). |

**Indexes**:
- `idx_doc_filename`: Optimizes queries filtering by filename.
- `idx_doc_strategy`: Optimizes queries filtering by splitting strategy.

## Usage

The project is designed to be run as a module. To index a document, you can run the script directly.

### Basic Usage
Index a document using the default 'fixed' chunking strategy:

```bash
python src/index_documents.py --file data/sample_document.pdf
```

### Advanced Usage
Specify a splitting strategy (`fixed`, `sentence`, or `paragraph`):

```bash
python src/index_documents.py --file data/sample_document.docx --strategy paragraph
```

### Help
View all available options:
```bash
python src/index_documents.py --help
```

### Command Line Arguments

| Argument | Type | Required | Default | Description | Choices |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `--file` | `str` | Yes | N/A | Path to the document file to be indexed (e.g., `.pdf`, `.docx`). | N/A |
| `--strategy` | `str` | No | `fixed` | The text splitting strategy to use. | `fixed`, `sentence`, `paragraph` |

## Testing

This project includes a comprehensive test suite using `pytest`.

To run the tests:
```bash
pytest
```

To run tests with verbose output:
```bash
pytest -v
```

## Project Structure

```mermaid
graph TD
    Root[semantic-document-indexer/]
    Root --> Src[src/]
    Root --> Data[data/]
    Root --> Tests[tests/]
    Root --> Env[.env]
    Root --> Req[requirements.txt]

    subgraph "src/"
        Src --> Indexer[index_documents.py]
        Src --> Loader[document_loader.py]
        Src --> Splitter[text_splitter.py]
        Src --> Embedder[embedding_client.py]
        Src --> DB[database_manager.py]
    end

    subgraph "tests/"
        Tests --> T_Indexer[test_index_documents.py]
        Tests --> T_Loader[test_document_loader.py]
        Tests --> T_Splitter[test_text_splitter.py]
        Tests --> T_Embedder[test_embedding_client.py]
        Tests --> T_DB[test_database_manager.py]
    end
```

### Key Modules
-   **`src/document_loader.py`**: Handles loading and cleaning of raw document files.
-   **`src/text_splitter.py`**: Implements different strategies for splitting text into chunks.
-   **`src/embedding_client.py`**: Interface for the Google GenAI embedding service.
-   **`src/database_manager.py`**: Manages PostgreSQL connections and vector data insertion.
-   **`src/index_documents.py`**: Main entry point that orchestrates the entire indexing workflow.
