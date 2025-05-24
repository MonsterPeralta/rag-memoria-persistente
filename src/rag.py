from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
import os
import logging
from typing import List

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        """Inicialización robusta del sistema RAG con manejo de errores"""
        try:
            # Configuración de ChromaDB
            self.chroma_path = os.path.abspath("./chroma_db")
            os.makedirs(self.chroma_path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.chroma_path)
            logger.info(f"✅ ChromaDB inicializado en: {self.chroma_path}")

            # Configuración de embeddings (forzando CPU)
            self.embeddings = HuggingFaceEmbeddings(
                model_name="paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("✅ Embeddings configurados correctamente")

            # Configuración del text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""]
            )

            self.vector_db = None

        except Exception as e:
            logger.error(f"❌ Error al inicializar RAGSystem: {str(e)}")
            raise

    def process_pdf(self, file_path: str) -> int:
        """Procesa un PDF y lo carga en la base de datos vectorial"""
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            chunks = self.text_splitter.split_documents(pages)
            
            self.vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                client=self.client,
                collection_name="pdf_documents",
                collection_metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"📄 Documento procesado: {len(chunks)} fragmentos")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"❌ Error al procesar PDF: {str(e)}")
            raise

    def query(self, question: str, k: int = 3) -> List[str]:
        """Consulta la base de datos vectorial"""
        if not self.vector_db:
            raise ValueError("Primero debes cargar un documento PDF.")
        return self.vector_db.similarity_search(question, k=k)