import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from mistralai import Mistral, UserMessage
import argparse

class MistralLLM:
    def __init__(self, api_key):
        self.client = Mistral(api_key=api_key)
        self.model = "mistral-large-latest"

    def call(self, prompt: str, stop=None):
        messages = [UserMessage(content=prompt)]
        response = self.client.chat.complete(model=self.model, messages=messages)
        return response.choices[0].message.content.strip()

def compute_embeddings(stories_folder, vector_db_path):
    stories_path = Path(stories_folder)
    if not stories_path.exists():
        print(f"Directory not found: {stories_folder}")
        return
    
    documents = []
    for file_path in stories_path.glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            documents.append(Document(page_content=content, metadata={"source": str(file_path)}))
    
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(documents, embedder, persist_directory=vector_db_path)
    vector_store.persist()
    print(f"Embeddings computed and saved to {vector_db_path}")

def get_character_info(character_name, vector_db_path):
    if not Path(vector_db_path).exists():
        print(f"Vector database not found: {vector_db_path}. Please run 'compute_embeddings' first.")
        return

    vector_store = Chroma(persist_directory=vector_db_path, embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    query = f"Provide structured details about the character '{character_name}' in JSON format. Include name, storyTitle, summary, relations, and characterType."
    related_docs = retriever.get_relevant_documents(query)
    matching_docs = [doc for doc in related_docs if character_name.lower() in doc.page_content.lower()]

    if not matching_docs:
        print(f"No documents containing character '{character_name}' were found.")
        return

    context = "\n\n".join([doc.page_content for doc in matching_docs])
    prompt = f"Context:\n{context}\n\n{query}"
    api_key = os.getenv("MISTRAL_API_KEY")
    mistral_llm = MistralLLM(api_key)
    response = mistral_llm.call(prompt)

    try:
        structured_response = json.loads(response)
        print(json.dumps(structured_response, indent=4))
    except json.JSONDecodeError:
        print(f"Could not parse response into JSON format. Response: {response}")

def main():
    parser = argparse.ArgumentParser(description="CLI tool for story embeddings and character information retrieval")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    compute_parser = subparsers.add_parser('compute-embeddings')
    compute_parser.add_argument('stories_folder', type=str, help='Folder containing story files')
    compute_parser.add_argument('vector_db_path', type=str, help='Path to save the vector database')

    info_parser = subparsers.add_parser('get-character-info')
    info_parser.add_argument('character_name', type=str, help='Character name to retrieve information for')
    info_parser.add_argument('vector_db_path', type=str, help='Path of the vector database')

    args = parser.parse_args()
    
    if args.command == 'compute-embeddings':
        compute_embeddings(args.stories_folder, args.vector_db_path)
    elif args.command == 'get-character-info':
        get_character_info(args.character_name, args.vector_db_path)

if __name__ == "__main__":
    load_dotenv()
    main()
