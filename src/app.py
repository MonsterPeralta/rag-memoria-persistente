import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
from rag import RAGSystem
from memory import PersistentChatMemory
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import warnings

# Configuración inicial
warnings.filterwarnings("ignore", message="Tried to instantiate class '__path__._path'")
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "chatbot-rag"

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Template del prompt
PROMPT_TEMPLATE = """
Responde basándote en este contexto:
{context}

Historial de chat:
{chat_history}

Pregunta: {input}
Respuesta:"""

def initialize_session_state():
    """Inicializa el estado de la sesión con manejo robusto"""
    try:
        # 1. Inicializar sistema RAG
        if "rag" not in st.session_state:
            st.session_state.rag = RAGSystem()
            logger.info("✅ Sistema RAG inicializado")

        # 2. Inicializar memoria persistente
        if "memory" not in st.session_state:
            mem_path = os.path.abspath("chat_memory.json")
            st.session_state.memory = PersistentChatMemory(path=mem_path)
            st.toast(f"💾 Memoria cargada desde: {mem_path}")
            logger.info(f"Memoria persistente inicializada en: {mem_path}")

        # 3. Inicializar mensajes de la interfaz
        if "messages" not in st.session_state:
            st.session_state.messages = []
            load_chat_history()

    except Exception as e:
        logger.error(f"❌ Error al inicializar sesión: {str(e)}")
        st.error("Error al inicializar la aplicación. Recarga la página.")

def load_chat_history():
    """Carga el historial de chat desde la memoria persistente"""
    try:
        # Cargar desde el archivo JSON directamente como respaldo
        if os.path.exists("chat_memory.json"):
            with open("chat_memory.json", "r", encoding="utf-8") as f:
                messages_data = json.load(f)
                
                for msg in messages_data:
                    if not isinstance(msg, dict):
                        continue
                        
                    content = msg.get("data", {}).get("content", msg.get("content", ""))
                    role = "user" if msg.get("type") == "human" else "assistant"
                    
                    if content and role:
                        st.session_state.messages.append({"role": role, "content": content})

        # Cargar desde la memoria de LangChain
        langchain_messages = st.session_state.memory.memory.chat_memory.messages
        for msg in langchain_messages:
            try:
                content = msg.content
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                if content and not any(m["content"] == content for m in st.session_state.messages):
                    st.session_state.messages.append({"role": role, "content": content})
            except Exception as e:
                logger.warning(f"⚠️ Error cargando mensaje de LangChain: {str(e)}")

    except Exception as e:
        logger.error(f"❌ Error cargando historial: {str(e)}")

def backup_message_to_json(user_input: str, assistant_output: str):
    """Guarda mensajes directamente en JSON como backup redundante"""
    backup_path = "chat_backup.json"
    try:
        messages = []
        
        # Cargar mensajes existentes
        if os.path.exists(backup_path):
            with open(backup_path, "r", encoding="utf-8") as f:
                messages = json.load(f)
        
        # Añadir nuevos mensajes
        messages.append({
            "type": "human",
            "content": user_input,
            "timestamp": str(datetime.now())
        })
        messages.append({
            "type": "ai",
            "content": assistant_output,
            "timestamp": str(datetime.now())
        })
        
        # Guardar
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2)
            
    except Exception as e:
        logger.error(f"⚠️ Error en backup JSON: {str(e)}")

def setup_sidebar():
    """Configura el panel lateral"""
    with st.sidebar:
        st.header("⚙️ Configuración del Modelo")
        st.slider("Temperatura", 0.0, 1.0, 0.7, 0.01, key="temperature")
        st.slider("Top-P", 0.1, 1.0, 0.9, 0.01, key="top_p")
        st.slider("Top-K", 1, 100, 50, key="top_k")

        if st.button("🧹 Limpiar memoria"):
            st.session_state.memory.clear()
            st.session_state.messages = []
            st.success("🧠 Memoria limpiada")
            st.rerun()

def main():
    """Función principal de la aplicación"""
    st.set_page_config(
        page_title="Chatbot RAG PDF", 
        page_icon="📄", 
        layout="centered"
    )
    st.title("📄 Chatbot RAG con Memoria Persistente")

    initialize_session_state()
    setup_sidebar()

    # Carga de PDF
    uploaded_file = st.file_uploader("Sube un archivo PDF", type="pdf")
    if uploaded_file:
        os.makedirs("data", exist_ok=True)
        doc_path = os.path.join("data", "documento.pdf")

        with open(doc_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            chunks = st.session_state.rag.process_pdf(doc_path)
            st.success(f"✅ Documento cargado y dividido en {chunks} fragmentos.")
        except Exception as e:
            st.error(f"❌ Error al procesar el PDF: {str(e)}")

    # Mostrar historial de mensajes
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Procesar entrada del usuario
    if prompt := st.chat_input("Haz una pregunta sobre el documento"):
        try:
            # Mostrar mensaje del usuario
            with st.chat_message("user"):
                st.markdown(prompt)

            # Obtener documentos relevantes
            docs = st.session_state.rag.query(prompt)

            # Configurar LLM
            llm = Ollama(
                model="llama3",
                temperature=st.session_state.temperature,
                top_p=st.session_state.top_p,
                top_k=st.session_state.top_k
            )

            # Crear y ejecutar cadena
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            chain = create_stuff_documents_chain(llm, prompt_template)

            response = chain.invoke({
                "input": prompt,
                "context": docs,
                "chat_history": st.session_state.memory.load_memory_variables({})
            })

            # Mostrar respuesta
            with st.chat_message("assistant"):
                st.markdown(response)

            # Guardar en memoria
            input_data = {"input": prompt}
            output_data = {"output": response}
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            st.session_state.memory.save_context(input_data, output_data)
            backup_message_to_json(prompt, response)

        except Exception as e:
            st.error(f"⚠️ Error al generar respuesta: {str(e)}")
            logger.error(f"Error en el flujo principal: {str(e)}")

if __name__ == "__main__":
    main()