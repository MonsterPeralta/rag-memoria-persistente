from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'}
)
result = embeddings.embed_documents(["test document"])
print("¡Funciona!" if len(result[0]) == 384 else "Error")