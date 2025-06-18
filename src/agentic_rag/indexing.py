# src/agentic_rag/indexing.py
# This script is responsible for the complete data ingestion and indexing pipeline.
# It performs a one-time task of reading a source document, processing its content
# using advanced vision-based OCR, chunking the cleaned text, generating vector
# embeddings, and finally loading everything into the Qdrant vector database.

import os
import asyncio
import uuid
from dotenv import load_dotenv

# --- Environment Variable Configuration ---
# This block is placed at the top-level of the module to ensure that environment
# variables are loaded and set *before* any other libraries (like pyzerox or litellm)
# are imported and attempt to access them. This is the most robust way to handle
# authentication for these libraries.

print("Loading environment variables from .env file...")
load_dotenv()

# Explicitly load and set the OPENAI_API_KEY. Zerox uses this key via its
# underlying 'litellm' dependency for its vision model API calls.
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file. It is required for Zerox vision processing.")
os.environ["OPENAI_API_KEY"] = openai_api_key
print("OPENAI_API_KEY has been set in the environment for Zerox to use.")


# --- Core Library Imports ---
# These are imported after the environment is configured.
from openai import OpenAI
from qdrant_client import QdrantClient, models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pyzerox import zerox

async def main():
    """
    The main asynchronous function that orchestrates the entire indexing workflow.
    """
    print("--- Starting High-Quality Indexing Pipeline ---")
    
    # --- Load Configuration ---
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME")
    
    if not all([qdrant_url, qdrant_collection_name]):
        raise ValueError("QDRANT_URL or QDRANT_COLLECTION_NAME is missing from .env file.")
    
    # Initialize the clients we'll need for this process.
    client_for_embedding = OpenAI(api_key=openai_api_key)
    qdrant_client = QdrantClient(url=qdrant_url)

    # --- Step 1: Prepare the Vector Database ---
    # We recreate the collection every time to ensure a fresh, clean start.
    # This prevents data duplication if the script is run multiple times.
    print(f"Recreating Qdrant collection: '{qdrant_collection_name}' with dimension 3072")
    qdrant_client.recreate_collection(
        collection_name=qdrant_collection_name,
        vectors_config=models.VectorParams(
            size=3072, # Dimension for OpenAI's text-embedding-3-large model
            distance=models.Distance.COSINE
        ),
    )

    # --- Step 2: Vision-Based Document Processing (Zerox) ---
    pdf_path = "documents/DBMS Notes.pdf"
    print(f"Processing '{pdf_path}' with Zerox...")
    
    try:
        # Call the async Zerox function. It converts each PDF page to an image
        # and sends it to the specified vision model to extract structured Markdown.
        zerox_result = await zerox(
            file_path=pdf_path,
            model="gpt-4o-mini" # A powerful and cost-effective vision model
        )
        
        # Combine the clean Markdown content from all pages into a single string.
        full_markdown_content = "\n\n".join([page.content for page in zerox_result.pages if page.content])
        print("Successfully extracted clean Markdown content.")
    except Exception as e:
        # Gracefully handle any errors from the Zerox/OpenAI API call.
        print(f"An error occurred during Zerox processing: {e}")
        # Provide helpful feedback for common quota/rate limit errors.
        if "rate_limit" in str(e).lower() or "quota" in str(e).lower():
            print("\n>>> This looks like an OpenAI quota or rate limit issue. Please check your OpenAI account billing details. <<<\n")
        import traceback
        traceback.print_exc()
        return

    # --- Step 3: Text Chunking ---
    # Split the clean Markdown content into smaller, manageable chunks.
    # The separators are chosen to respect Markdown formatting (paragraphs, headers).
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "##", "#", " "],
        chunk_size=1024,
        chunk_overlap=150
    )
    chunks = text_splitter.split_text(full_markdown_content)
    print(f"Split content into {len(chunks)} chunks.")

    # --- Step 4: Embedding and Upserting ---
    print("Embedding chunks with OpenAI and uploading to Qdrant...")
    embedding_model = "text-embedding-3-large"
    batch_size = 100 # Process chunks in batches for API efficiency.
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        
        # Generate vector embeddings for the batch of chunks.
        response = client_for_embedding.embeddings.create(input=batch_chunks, model=embedding_model)
        embeddings = [record.embedding for record in response.data]
        
        # Prepare the data points for Qdrant, including the vector and the original text as payload.
        qdrant_client.upsert(
            collection_name=qdrant_collection_name,
            points=[
                models.PointStruct(id=str(uuid.uuid4()), vector=embedding, payload={"text": chunk})
                for embedding, chunk in zip(embeddings, batch_chunks)
            ],
            wait=True, # Ensure the operation is confirmed before proceeding.
        )
        print(f"Uploaded batch {i//batch_size + 1}.")

    # --- Final Verification ---
    print("\n--- High-Quality Indexing Complete ---")
    stats = qdrant_client.get_collection(collection_name=qdrant_collection_name).vectors_count
    print(f"Collection '{qdrant_collection_name}' now contains {stats} high-quality vectors.")

if __name__ == "__main__":
    # The script entry point, using asyncio to run our main async function.
    asyncio.run(main())