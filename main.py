import os
import fitz  # PyMuPDF
from langchain_text_splitters import CharacterTextSplitter
from google import genai

client = genai.Client(api_key="AIzaSyARLvQFQmaMr5TwXMuiS-AD--3O9NMCzOs")

def load_pdf_from_user():
    pdf_path = input("Enter the path to your PDF file: ").strip()

    # Check if file exists
    if not os.path.exists(pdf_path):
        print("❌ File not found. Please check the path.")
        return None

    # Open PDF
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
    print("pages data",pages_data)
    return pages_data

def chunk_pdf_pages(pages_data):
    # Create text splitter  
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=100,
        chunk_overlap=0
    )


    chunks_with_metadata = []

    for page in pages_data:
        # Split text for each page
        chunks = text_splitter.split_text(page["content"])
        
        # Add page metadata to each chunk
        for chunk in chunks:
            chunks_with_metadata.append({
                "page_number": page["page_number"],
                "content": chunk
            })

    return chunks_with_metadata

def create_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        result = client.models.embed_content(
            model="models/gemini-embedding-001",
            content=chunk["content"]
        )
        embeddings.append({
            "page_number": chunk["page_number"],
            "content": chunk["content"],
            "embedding": result.embedding.values
        })
    return embeddings


if __name__ == "__main__":
    pages = load_pdf_from_user()

    if pages:
        chunks = chunk_pdf_pages(pages)

        print("\n=== Chunks Preview ===\n")
        for c in chunks[:5]:  # Show first 5 chunks
            print(f"Page {c['page_number']} → {c['content']}")
            print("-" * 50)


        vector_embeddings = create_embeddings(chunks)
        print("\n✅ Created embeddings for all chunks!")
        print(f"Example embedding (length {len(vector_embeddings[0]['embedding'])}):")
        print(vector_embeddings[0])
