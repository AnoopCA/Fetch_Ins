import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_classic.prompts import PromptTemplate
from langchain_classic.schema import HumanMessage

def process_all_PDFs(PDF_Directory):
    all_documents = []
    pdf_path = Path(PDF_Directory)
    pdf_files = list(pdf_path.glob("**/*.pdf"))

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()
        for doc in documents:
            doc.metadata['source_file'] = pdf_file.name
            doc.metadata['file_type'] = "pdf"
        all_documents.extend(documents)

    print(f"Total documents loaded:{(len(all_documents))}")
    return all_documents

def split_documents(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        separators = ["\\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, texts):
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings

class VectorStore:
    def __init__(self, collection_name, persistent_directory):
        os.makedirs(persistent_directory, exist_ok = True)
        self.client = chromadb.PersistentClient(path = persistent_directory)
        self.collection = self.client.get_or_create_collection(
            name = collection_name,
            metadata = {"Description" : "PDF documents embedding for RAG"}
        )

    def add_document(self, documents, embeddings):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        for i, (doc, embed) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata['doc_idx'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embed.tolist())

        self.collection.add(
            ids = ids,
            embeddings = embeddings_list,
            metadatas = metadatas,
            documents = documents_text
        )

        print(f"Successfully added {len(documents)} to the vector store")
        print(f"Total documents in collection: {self.collection.count()}")

class RAG_Retriever:
    def __init__(self, vector_store, embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query, top_k, score_threshold = 0.3):
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)
        retrieved_docs = []
        if results['documents'] and results['documents'][0]:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            ids = results['ids'][0]
            distances = results['distances'][0]
            for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                similarity = 1- distance
                if similarity >= score_threshold:
                    retrieved_docs.append({
                        'id' : doc_id,
                        'content' : document,
                        'metadata' : metadata,
                        'similarity' : similarity,
                        'distance' : distance,
                        'rank' : i+1
                    })
            print(f'Retrieved {len(retrieved_docs)} documents after filtering')
        else:
            print('No documents retrieved')
        return retrieved_docs

class GroqLLM:
    def __init__(self, model_name, api_key):
        self.llm = ChatGroq(groq_api_key=api_key, model_name=model_name, temperature=0.1, max_tokens=1024)
        self.prompt_template = PromptTemplate(
                    input_variables=["context", "question"],
                    template= """You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.
                    Context: {context}
                    Question: {question}
                    Answer instructions:
                    - If the answer is fully supported by the context, provide it directly.
                    Answer:"""
                )
                    #- If the context does not provide enough information, answer based on your knowledge, and explicitly start your answer with: "Note: The answer below is based on general knowledge, not the provided context."
                
    def generate_response(self, query, context):
        # Format the prompt
        formatted_prompt = self.prompt_template.format(context=context, question=query)
        # Generate response
        messages = [HumanMessage(content=formatted_prompt)]
        response = self.llm.invoke(messages)
        return response.content

if __name__ == "__main__":
    load_dotenv(r"D:\ML_Projects\Fetch_Ins\Keys\.env")
    groq_api_key = os.getenv("GROQ_API_KEY")

    all_pdf_docs = process_all_PDFs(r"D:\ML_Projects\Fetch_Ins\Policy Documents")
    chunks = split_documents(all_pdf_docs, chunk_size=1000, chunk_overlap=100)
    embeddingManager = EmbeddingManager()
    vector_store = VectorStore(collection_name = "PDF_Documents", persistent_directory=r"D:\ML_Projects\Fetch_Ins\Vector Store\chromadb")
    texts = [doc.page_content for doc in chunks]
    embeddings = embeddingManager.generate_embeddings(texts)
    vector_store.add_document(chunks, embeddings)
    retriever = RAG_Retriever(vector_store, embeddingManager)

    groq_llm=GroqLLM(model_name="openai/gpt-oss-20b", api_key=groq_api_key)

    while True:
        query = input("Enter the question:\n")
        if query.lower() == "exit":
            break
        results = retriever.retrieve(query, top_k = 5)
        context = "\n\n".join([doc['content'] for doc in results]) if results else ""
        answer=groq_llm.generate_response(query, context)
        print(answer)
