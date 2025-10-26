# -----------------------------
# Imports for RAG Integration
# -----------------------------
from RetrievalMind.embeddings_manager import EmbeddingManager
from RetrievalMind.data_ingestion import PDFDocumentIngestor
import chromadb
import uuid
import logging
logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# Imports for AI Agent
# -----------------------------
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory.summary import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
import os
# Allow enabling verbose debug prints by setting environment variable RAG_DEBUG=1
if os.getenv("RAG_DEBUG", "0") == "1":
    logger.setLevel(logging.DEBUG)

# -----------------------------
# Document Loading and Embedding
# -----------------------------
def load_pdf_document(pdf_folder_path: str) -> list:
    """
    Load all PDF documents from a folder and return them as chunks.

    Args:
        pdf_folder_path (str): Path to the folder containing PDFs.
        file_pattern (str): Glob pattern to match PDF files (default '**/*.pdf').

    Returns:
        list: List of document chunks with page content and metadata.
    """
    pdf_ingestor = PDFDocumentIngestor(file_path = pdf_folder_path, loader_type='mu')
    pdf_loader = pdf_ingestor.load_document()
    document_chunks = pdf_loader.load()
    logger.debug(f"Loaded {len(document_chunks)} document chunks from {pdf_folder_path}")

    # Debug: preview first 5 document chunks
    for idx, chunk in enumerate(document_chunks[:5]):
        logger.debug(f"Document Chunk {idx} preview: {repr(chunk.page_content[:100])}")
    
    return document_chunks


def generate_document_embeddings(document_chunks: list, embedding_model: str = "all-miniLM-L6-v2") -> tuple:
    """
    Generate embeddings for a list of document chunks using a SentenceTransformer model.

    Args:
        document_chunks (list): List of document chunks.
        embedding_model (str): Name of the embedding model (default "all-miniLM-L6-v2").

    Returns:
        tuple: embeddings list, EmbeddingManager instance
    """
    texts = [chunk.page_content for chunk in document_chunks]
    embedding_manager = EmbeddingManager(model_name=embedding_model)
    embeddings = embedding_manager.generate_embeddings(texts)
    logger.debug(f"Generated embeddings for {len(texts)} chunks using model '{embedding_model}'")
    return embeddings, embedding_manager


def store_documents_in_vector_store(document_chunks: list, embeddings: list, collection_name: str, persist_dir: str, doc_type: str = "PDF") -> object:
    """
    Store document chunks and embeddings in a ChromaDB vector store.

    Args:
        document_chunks (list): List of document chunks.
        embeddings (list): Corresponding embeddings for each chunk.
        collection_name (str): Name of the vector store collection.
        persist_dir (str): Directory where the vector store is persisted.
        doc_type (str): Type of documents being stored (default "PDF").

    Returns:
        VectorStore: Initialized and populated vector store instance.
    """
    # Ensure persistence directory exists
    os.makedirs(persist_dir, exist_ok=True)

    # Initialize persistent ChromaDB client and collection
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": f"{doc_type} document embeddings for RAG pipeline"}
    )

    # Prepare records for ChromaDB
    ids, metadatas, documents_text, embeddings_list = [], [], [], []
    for i, (doc, emb) in enumerate(zip(document_chunks, embeddings)):
        doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
        ids.append(doc_id)

        metadata = dict(getattr(doc, "metadata", {}))
        metadata.update({
            "doc_index": i,
            "content_length": len(getattr(doc, "page_content", "")),
            "source": getattr(doc, "metadata", {}).get("source", None)
        })
        metadatas.append(metadata)

        documents_text.append(getattr(doc, "page_content", ""))
        embeddings_list.append(emb.tolist() if hasattr(emb, 'tolist') else list(emb))

    collection.add(
        ids=ids,
        documents=documents_text,
        metadatas=metadatas,
        embeddings=embeddings_list
    )

    logger.debug(f"Successfully added {len(documents_text)} documents to '{collection_name}'.")
    logger.debug(f"Total documents in collection: {collection.count()}")
    return collection


def query_vector_store(collection: object, embedding_manager: EmbeddingManager, query_text: str, top_k: int = 3, min_score: float = 0.0, raw_documents: list | None = None) -> list:
    """
    Perform semantic search on the vector store.

    Args:
        vector_store (VectorStore): Initialized vector store instance.
        embedding_manager (EmbeddingManager): Embedding generator.
        query_text (str): Query string for retrieval.
        top_k (int): Number of top results to return.
        min_score (float): Minimum similarity score threshold.

    Returns:
        list: List of retrieved documents with metadata and similarity scores.
    """
    # Direct Chroma query using the provided collection
    results = []
    try:
        try:
            q_embs = embedding_manager.generate_embeddings([query_text])
            q_emb = q_embs[0] if hasattr(q_embs, '__len__') and len(q_embs) > 0 else q_embs
            logger.debug(f"Query embedding length: {getattr(q_emb, 'shape', None) or len(q_emb)}")
            logger.debug(f"Query embedding (first 6 values): {q_emb[:6]}")
        except Exception as e:
            logger.debug(f"Failed to generate query embedding: {e}")
            raise

        raw = collection.query(query_embeddings=[q_emb.tolist()], n_results=top_k)
        raw_dist = raw.get('distances', [[]])[0] if raw.get('distances') else []
        raw_docs = raw.get('documents', [[]])[0] if raw.get('documents') else []
        raw_ids = raw.get('ids', [[]])[0] if raw.get('ids') else []
        raw_metas = raw.get('metadatas', [[]])[0] if raw.get('metadatas') else []

        interpreted = []
        for idx, d in enumerate(raw_dist):
            try:
                similarity = 1.0 / (1.0 + float(d))
            except Exception:
                similarity = 0.0

            if similarity >= min_score and idx < len(raw_docs):
                interpreted.append({
                    'id': raw_ids[idx] if idx < len(raw_ids) else f'raw-{idx}',
                    'content': raw_docs[idx],
                    'metadata': raw_metas[idx] if idx < len(raw_metas) else {},
                    'similarity_score': round(similarity, 4),
                    'distance': round(d, 4)
                })

        logger.info(f"Direct Chroma reinterpretation returned {len(interpreted)} results")
        results = interpreted
    except Exception as e:
        logger.debug(f"Direct Chroma query failed: {e}")

    # If still no results or results don't answer a teacher-related query, attempt a keyword fallback over raw documents
    lower_q = query_text.lower()
    teacher_query = any(k in lower_q for k in ['teacher', 'teach', 'instructor', 'who are the teachers'])

    # If no results, or this is a teacher query and top results don't contain teacher info, run fallback
    need_fallback = (len(results) == 0 and raw_documents) or (
        teacher_query and raw_documents and not any(
            any(tok in (r.get('content') or '').lower() for tok in ['teacher', 'instructor', 'himanshu', 'mihir', 'shivam'])
            for r in results
        )
    )

    if need_fallback:
        logger.warning("Running keyword fallback search over raw documents (teacher-related or no direct hits).")
        fallback = []
        for idx, doc in enumerate(raw_documents):
            text = getattr(doc, 'page_content', None) or (doc.get('page_content') if isinstance(doc, dict) else str(doc))
            if not text:
                continue
            if any(k in text.lower() for k in ['teacher', 'instructor', 'himanshu', 'mihir', 'shivam']):
                fallback.append({
                    'id': f'fallback-{idx}',
                    'similarity_score': 1.0,
                    'content': text,
                })
        logger.info(f"Fallback matched {len(fallback)} documents")
        if fallback:
            return fallback

    return results

# -----------------------------
# RAG + LLM Integration
# -----------------------------
def initialize_llm(api_key_env_var: str = "Gemini_APIKEY") -> GoogleGenerativeAI:
    """
    Initialize the Google Gemini LLM using the API key from environment variables.
    """
    api_key = os.getenv(api_key_env_var)
    if not api_key:
        raise ValueError(f"API key not found in environment variable '{api_key_env_var}'")
    return GoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)


def get_prompt() -> ChatPromptTemplate:
    """
    Returns a ChatPromptTemplate for the RAG AI assistant specialized in company policies.
    """
    return ChatPromptTemplate([
        ('system', """You are a friendly and knowledgeable AI assistant for an online education company.
        Your job is to answer student or visitor questions about classes, teachers, schedules, pricing, and ongoing batches.
        Always maintain a warm, helpful, and professional tone suitable for an educational environment."""),

        ('user', """Here are the retrieved company information documents relevant to the user’s question:\n
        {retrieved_docs_from_rag}\n
        User Query: {user_query}\n
        Instructions:

        * Answer ONLY based on the details available in the documents above.
        * If the answer is not found, politely inform the user that the information is currently unavailable and suggest contacting support. 
          You can say: "Please feel free to contact our support team directly. They would be happy to assist you! +91 91233 66161"
        * Keep the answer short, clear, and friendly.
        * Reference teacher names, course timings, or pricing details when relevant.
        * Do NOT generate unrelated or fabricated information.
        Provide your answer below:""")

    ])


def get_chain(llm: GoogleGenerativeAI, prompt_template: ChatPromptTemplate) -> Runnable:
    """
    Create a Runnable chain combining the prompt template, LLM, and output parser.
    """
    return prompt_template | llm | StrOutputParser()


def ask_with_rag(chain: Runnable, query: str, retrieved_docs: list) -> str:
    """
    Generate an AI answer for a query based on RAG retrieved documents.
    """
    docs_text = "\n".join([doc['content'] for doc in retrieved_docs])
    input_mapping = {
        "user_query": query,
        "retrieved_docs_from_rag": docs_text
    }
    return chain.invoke(input_mapping)

# -----------------------------
# Main Execution
# -----------------------------
def main(query: str):
    """
    Full RAG pipeline: load documents, generate embeddings, store/retrieve, and prepare for LLM query.
    """
    # Configuration
    pdf_folder = "backend/data/pdf/Company_Details.pdf"
    vector_collection_name = "policy_pal_vector_collection"
    vector_store_directory = "backend/data/policy_pal_vector_store"

    # Step 1: Load PDF documents (example: "Travel Policy", "Expense Policy", "HR Guidelines")
    pdf_chunks = load_pdf_document(pdf_folder)

    # Quick keyword scan for teacher-related queries to ensure exact section hits
    lower_q = query.lower()
    if any(k in lower_q for k in ['teacher', 'teach', 'instructor', 'who are the teachers']):
        teacher_matches = []
        for idx, chunk in enumerate(pdf_chunks):
            text = getattr(chunk, 'page_content', '')
            if not text:
                continue
            if any(tok in text.lower() for tok in ['teacher', 'instructor', 'himanshu', 'mihir', 'shivam']):
                teacher_matches.append({
                    'id': f'chunk-{idx}',
                    'content': text,
                    'metadata': dict(getattr(chunk, 'metadata', {})),
                    'similarity_score': 1.0
                })

        if teacher_matches:
            logger.info(f"Found {len(teacher_matches)} teacher-related document chunks via keyword scan.")
            return teacher_matches

    # Step 2: Generate embeddings
    embeddings, embedding_manager = generate_document_embeddings(pdf_chunks)

    # Step 3: Store documents in vector store (Chroma collection)
    collection = store_documents_in_vector_store(pdf_chunks, embeddings, vector_collection_name, vector_store_directory)

    # Step 4: Retrieve relevant documents
    # --- DEBUG: run a direct ChromaDB query with the generated query embedding ---
    try:
        q_embs = embedding_manager.generate_embeddings([query])
        # Handle numpy array or list returns safely
        q_emb = q_embs[0] if hasattr(q_embs, '__len__') and not isinstance(q_embs, float) else q_embs
        logger.debug(f"Direct query embedding shape/len: {getattr(q_emb, 'shape', None) or len(q_emb)}")
        raw_query_result = collection.query(query_embeddings=[q_emb.tolist()], n_results=5)
        logger.debug("Raw Chroma query result keys: %s", list(raw_query_result.keys()))
        # Print summaries of returned sections
        for k, v in raw_query_result.items():
            try:
                logger.debug("%s: type=%s len=%s", k, type(v), (len(v) if hasattr(v, '__len__') else 'N/A'))
            except Exception:
                logger.debug("%s: (unprintable)", k)
        logger.debug("Raw query result (sample): %s", {k: (v[:1] if isinstance(v, list) else str(v)) for k, v in raw_query_result.items()})
    except Exception as e:
        logger.error(f"Failed to run direct chroma query for debug: {e}")

    # Pass raw_documents so our fallback can operate if retriever returns nothing
    retrieved_docs = query_vector_store(collection, embedding_manager, query_text=query, top_k=5, min_score=0.0, raw_documents=pdf_chunks)

    # Preview retrieved docs (debug only)
    for doc in retrieved_docs:
        logger.debug("Retrieved Document ID: %s", doc.get('id'))
        logger.debug("Similarity Score: %s", doc.get('similarity_score'))
        logger.debug("Content Preview: %s...", (doc.get('content') or '')[:500])
    return retrieved_docs


if __name__ == "__main__":
    query_text = "What courses are currently available for enrollment?"

    # Step 1-4: Run RAG pipeline
    retrieved_docs = main(query=query_text)
    logger.debug("retrieved_docs: %s", retrieved_docs)

    # Step 5: Initialize LLM
    llm_model = initialize_llm()

    # Step 6: Prepare chain
    prompt_template = get_prompt()
    rag_chain = get_chain(llm_model, prompt_template)

    # Step 7: Ask query
    answer = ask_with_rag(rag_chain, query=query_text, retrieved_docs=retrieved_docs)
    # Print only the AI answer to stdout for CLI usage
    print(answer)