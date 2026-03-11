from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def open_vector_db(chunks) -> Chroma:
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={'device': 'cuda'},  # Use 'cpu' if no GPU
        encode_kwargs={'normalize_embeddings': True}
    )

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db_ai_strategy"  # Save locally
    )
    print("Vector store created and persisted.")

    return vectorstore
