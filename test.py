try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.llms import Ollama
    print("¡Todos los imports funcionan correctamente!")
except ImportError as e:
    print(f"Error: {e}")