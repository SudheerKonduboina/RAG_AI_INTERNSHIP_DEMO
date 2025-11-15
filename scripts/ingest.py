import os
import argparse
import json
from PyPDF2 import PdfReader
from utils.helpers import clean_text, chunk_text


def load_pdfs_from_folder(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing: {pdf_path}")

            try:
                reader = PdfReader(pdf_path)
                text = ""

                for page in reader.pages:
                    text += page.extract_text() or ""

                documents.append({
                    "filename": filename,
                    "content": text
                })

            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Folder containing PDF files")
    parser.add_argument("--out", required=True, help="Output JSON file")
    args = parser.parse_args()

    folder_path = args.source

    if not os.path.isdir(folder_path):
        raise ValueError("Source must be a folder containing PDF files")

    docs = load_pdfs_from_folder(folder_path)

    all_chunks = []
    for doc in docs:
        cleaned = clean_text(doc["content"])
        chunks = chunk_text(cleaned)
        all_chunks.extend(chunks)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nSaved processed docs to: {args.out}")


if __name__ == "__main__":
    main()
