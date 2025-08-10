import os
import fitz  # PyMuPDF
from langchain_text_splitters import CharacterTextSplitter
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from dotenv import load_dotenv

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

# Google GenAI API client
client = genai.Client(api_key= google_api_key)

# Qdrant client (local instance)
qdrant = QdrantClient(url="http://localhost:6333")
collection_name = "book_embeddings"

# Create Qdrant collection if not exists
def create_qdrant_collection(vector_size):
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(
            size=vector_size,
            distance=rest.Distance.COSINE
        )
    )
    print(f"✅ Qdrant collection '{collection_name}' created!")

# Load PDF file from user input
def load_pdf_from_user():
    pdf_path = input("Enter the path to your PDF file: ").strip()
    if not os.path.exists(pdf_path):
        print("❌ File not found. Please check the path.")
        return None

    pdf_document = fitz.open(pdf_path)
    pages_data = []
    print(f"✅ Loaded PDF: {pdf_path}")
    print(f"📄 Total Pages: {len(pdf_document)}\n")

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text = page.get_text()
        pages_data.append({
            "page_number": page_num + 1,
            "content": text.strip()
        })

    pdf_document.close()
    return pages_data

# Split PDF pages into chunks
def chunk_pdf_pages(pages_data):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=100,
        chunk_overlap=0
    )
    chunks_with_metadata = []
    for page in pages_data:
        chunks = text_splitter.split_text(page["content"])
        for chunk in chunks:
            chunks_with_metadata.append({
                "page_number": page["page_number"],
                "content": chunk
            })
    return chunks_with_metadata

# Create embeddings and store in Qdrant
def create_and_store_embeddings(chunks):
    embeddings = []
    for i, chunk in enumerate(chunks):
        result = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=chunk["content"]
        )
        vector = result.embeddings[0].values

        embeddings.append({
            "page_number": chunk["page_number"],
            "content": chunk["content"],
            "embedding": vector
        })

        # Store in Qdrant
        qdrant.upsert(
            collection_name=collection_name,
            points=[
                rest.PointStruct(
                    id=i,
                    vector=vector,
                    payload={
                        "page_number": chunk["page_number"],
                        "content": chunk["content"]
                    }
                )
            ]
        )

    return embeddings


if __name__ == "__main__":
    pages = load_pdf_from_user()
    if pages:
        chunks = chunk_pdf_pages(pages)
        print(f"✂️ Split into {len(chunks)} chunks.")

        # Create Qdrant collection (size from a test embedding)
        test_result = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents="test"
        )
        vector_size = len(test_result.embeddings[0].values)
        create_qdrant_collection(vector_size)

        # Create embeddings & store in Qdrant
        vector_embeddings = create_and_store_embeddings(chunks)
        print(f"\n✅ Stored {len(vector_embeddings)} embeddings in Qdrant!")
        print(f"Example: {vector_embeddings[0]}")
