# src/indexing.py
# The definitive script to process PDFs with Zerox and load into Qdrant.

import os
import asyncio
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient, models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pyzerox import zerox

async def main():
    """Main function to run the Zerox indexing pipeline."""
    print("--- Starting High-Quality Indexing Pipeline ---")
    load_dotenv()

    # Load configuration
    openai_api_key = os.getenv("OPENAI_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME")
    
    if not all([openai_api_key, qdrant_url, qdrant_collection_name]):
        raise ValueError("One or more required environment variables are missing.")
    # Set the key for Zerox/LiteLLM to find
    os.environ["OPENAI_API_KEY"] = openai_api_key

    qdrant_client = QdrantClient(url=qdrant_url)
    openai_client = OpenAI(api_key=openai_api_key)

    # Recreate the collection for a fresh start
    print(f"Recreating Qdrant collection: '{qdrant_collection_name}'")
    qdrant_client.recreate_collection(
        collection_name=qdrant_collection_name,
        vectors_config=models.VectorParams(size=3072, distance=models.Distance.COSINE),
    )

    # Process the PDF using Zerox vision model
    pdf_path = "documents/DBMS Notes.pdf"
    print(f"Processing '{pdf_path}' with Zerox...")
    try:
        zerox_result = await zerox(file_path=pdf_path, model="gpt-4o-mini")
        full_markdown_content = "\n\n".join([page.content for page in zerox_result.pages if page.content])
        print("Successfully extracted clean Markdown content.")
    except Exception as e:
        print(f"An error occurred during Zerox processing: {e}")
        return

    # Chunk the clean content
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "##", "#", " "], chunk_size=1024, chunk_overlap=150
    )
    chunks = text_splitter.split_text(full_markdown_content)
    print(f"Split content into {len(chunks)} chunks.")

    # Embed and upload to Qdrant
    print("Embedding chunks and uploading to Qdrant...")
    embedding_model = "text-embedding-3-large"
    for i in range(0, len(chunks), 100):
        batch_chunks = chunks[i:i + 100]
        response = openai_client.embeddings.create(input=batch_chunks, model=embedding_model)
        
        qdrant_client.upsert(
            collection_name=qdrant_collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=record.embedding,
                    payload={"text": chunk}
                ) for record, chunk in zip(response.data, batch_chunks)
            ],
            wait=True,
        )
        print(f"Uploaded batch {i//100 + 1}.")

    print("\n--- Indexing Complete ---")
    stats = qdrant_client.get_collection(collection_name=qdrant_collection_name).vectors_count
    print(f"Collection '{qdrant_collection_name}' now contains {stats} high-quality vectors.")

if __name__ == "__main__":
    asyncio.run(main())