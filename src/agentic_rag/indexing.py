# src/agentic_rag/indexing.py
# V5 - DEFINITIVE FIX for Zerox/LiteLLM Authentication by setting env var at module level.

import os
import asyncio
import uuid
from dotenv import load_dotenv

# --- THIS IS THE FIX ---
# Load environment variables at the top-level of the module.
# This ensures they are set before any other library (like zerox) is imported and used.
print("Loading environment variables from .env file...")
load_dotenv()

# Set the OPENAI_API_KEY environment variable for LiteLLM/Zerox to find.
# This is the most reliable way to provide the key.
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file. It is required for Zerox vision processing.")
os.environ["OPENAI_API_KEY"] = openai_api_key
print("OPENAI_API_KEY has been set in the environment for Zerox to use.")
# --- END OF FIX ---


# Now, import the rest of the libraries that might depend on the environment
from openai import OpenAI
from qdrant_client import QdrantClient, models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pyzerox import zerox

async def main():
    """Main function to run the Zerox indexing pipeline."""
    print("--- Starting High-Quality Indexing Pipeline ---")
    
    # Load other variables within the function
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME")
    
    if not all([qdrant_url, qdrant_collection_name]):
        raise ValueError("QDRANT_URL or QDRANT_COLLECTION_NAME is missing from .env file.")
    
    # We still need an OpenAI client for the final embedding step
    client_for_embedding = OpenAI(api_key=openai_api_key)
    qdrant_client = QdrantClient(url=qdrant_url)

    # Recreate the collection with the correct dimensions for OpenAI embeddings
    print(f"Recreating Qdrant collection: '{qdrant_collection_name}' with dimension 3072")
    qdrant_client.recreate_collection(
        collection_name=qdrant_collection_name,
        vectors_config=models.VectorParams(size=3072, distance=models.Distance.COSINE),
    )

    pdf_path = "documents/DBMS Notes.pdf"
    print(f"Processing '{pdf_path}' with Zerox...")
    
    try:
        # We can now call zerox directly, as it will find the key in the environment
        zerox_result = await zerox(
            file_path=pdf_path,
            model="gpt-4o-mini"
        )
        
        full_markdown_content = "\n\n".join([page.content for page in zerox_result.pages if page.content])
        print("Successfully extracted clean Markdown content.")
    except Exception as e:
        print(f"An error occurred during Zerox processing: {e}")
        # Add more detail to the error message for debugging
        if "rate_limit" in str(e).lower() or "quota" in str(e).lower():
            print("\n>>> This looks like an OpenAI quota or rate limit issue. Please check your OpenAI account billing details. <<<\n")
        import traceback
        traceback.print_exc()
        return

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "##", "#", " "], chunk_size=1024, chunk_overlap=150
    )
    chunks = text_splitter.split_text(full_markdown_content)
    print(f"Split content into {len(chunks)} chunks.")

    print("Embedding chunks with OpenAI and uploading to Qdrant...")
    embedding_model = "text-embedding-3-large"
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        response = client_for_embedding.embeddings.create(input=batch_chunks, model=embedding_model)
        embeddings = [record.embedding for record in response.data]
        
        qdrant_client.upsert(
            collection_name=qdrant_collection_name,
            points=[
                models.PointStruct(id=str(uuid.uuid4()), vector=embedding, payload={"text": chunk})
                for embedding, chunk in zip(embeddings, batch_chunks)
            ],
            wait=True,
        )
        print(f"Uploaded batch {i//batch_size + 1}.")

    print("\n--- High-Quality Indexing Complete ---")
    stats = qdrant_client.get_collection(collection_name=qdrant_collection_name).vectors_count
    print(f"Collection '{qdrant_collection_name}' now contains {stats} high-quality vectors.")

if __name__ == "__main__":
    asyncio.run(main())