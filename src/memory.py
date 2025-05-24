import os
import json
from typing import List, Dict, Any, Optional
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import messages_from_dict, message_to_dict
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.base import BaseMessage
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersistentChatMemory:
    def __init__(self, path: Optional[str] = None):
        """Inicializa memoria persistente con manejo robusto de errores"""
        self.path = os.path.abspath(path) if path else os.path.join(
            os.getcwd(), "chat_memory.json"
        )
        
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                input_key="input",
                output_key="output"
            )
            
            self._verify_and_load()
            logger.info(f"💾 Memoria persistente inicializada en: {self.path}")
            
        except Exception as e:
            logger.error(f"❌ Error al inicializar memoria: {str(e)}")
            raise

    def _verify_and_load(self) -> None:
        """Carga y valida el archivo de memoria"""
        if not os.path.exists(self.path):
            logger.info("ℹ️ No se encontró archivo de memoria. Se creará uno nuevo.")
            return
            
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                raise ValueError("Formato inválido: debe ser lista")
                
            validated_messages = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                    
                # Extraer contenido y tipo de forma robusta
                content = item.get("data", {}).get("content", item.get("content", "")).strip()
                msg_type = item.get("type", "")
                
                if not content:
                    continue
                    
                if msg_type == "human" or "HumanMessage" in str(item):
                    validated_messages.append(HumanMessage(content=content))
                elif msg_type == "ai" or "AIMessage" in str(item):
                    validated_messages.append(AIMessage(content=content))
            
            # Asignar todos los mensajes válidos de una vez
            if validated_messages:
                self.memory.chat_memory.messages = validated_messages
                logger.info(f"📖 Cargados {len(validated_messages)} mensajes válidos")
                
        except Exception as e:
            logger.error(f"⚠️ Error al cargar memoria: {str(e)}")
            self._backup_and_reset()

    def _validate_messages(self, raw_messages: List[Dict[str, Any]]) -> List[BaseMessage]:
        """Valida y filtra mensajes crudos"""
        validated = []
        
        for idx, item in enumerate(raw_messages, 1):
            if not isinstance(item, dict):
                logger.warning(f"⚠️ Entrada {idx}: No es un diccionario, omitiendo")
                continue
                
            msg_type = item.get("type")
            content = item.get("content", "").strip()
            
            if not content:
                logger.warning(f"⚠️ Entrada {idx}: Mensaje sin contenido, omitiendo")
                continue
                
            try:
                if msg_type == "human":
                    validated.append(HumanMessage(content=content))
                elif msg_type == "ai":
                    validated.append(AIMessage(content=content))
                else:
                    logger.warning(f"⚠️ Entrada {idx}: Tipo de mensaje desconocido '{msg_type}'")
            except Exception as e:
                logger.error(f"⚠️ Entrada {idx}: Error al crear mensaje - {str(e)}")
                
        return validated

    # Resto de los métodos permanecen igual...
    def _backup_and_reset(self) -> None:
        """Crea backup y reinicia memoria"""
        backup_path = f"{self.path}.bak"
        try:
            if os.path.exists(self.path):
                os.rename(self.path, backup_path)
                logger.warning(f"⚠️ Creando backup en: {backup_path}")
        except Exception as e:
            logger.error(f"❌ Error en backup: {str(e)}")
            
        self.memory.clear()

    def save_context(self, input_data: Dict[str, Any], output_data: Dict[str, Any]) -> None:
        """Guarda el contexto con validación"""
        try:
            if not isinstance(input_data, dict) or not isinstance(output_data, dict):
                raise TypeError("Los datos deben ser diccionarios")
                
            if not input_data.get("input") or not output_data.get("output"):
                raise ValueError("Los datos deben contener 'input' y 'output' no vacíos")
                
            self.memory.save_context(input_data, output_data)
            self._atomic_save()
        except Exception as e:
            logger.error(f"⚠️ Error al guardar contexto: {str(e)}")
            raise

    def _atomic_save(self) -> None:
        """Guarda atómicamente usando patrón tempfile"""
        temp_path = f"{self.path}.tmp"
        try:
            messages = [m for m in self.memory.chat_memory.messages if getattr(m, "content", None)]
            data = [message_to_dict(m) for m in messages]
            
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            if os.path.exists(self.path):
                os.remove(self.path)
            os.rename(temp_path, self.path)
            
        except Exception as e:
            logger.error(f"❌ Error crítico al guardar: {str(e)}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            raise

    def load_memory_variables(self, inputs: Optional[Dict] = None) -> Dict[str, Any]:
        """Carga variables de memoria"""
        try:
            return self.memory.load_memory_variables(inputs or {})
        except Exception as e:
            logger.error(f"⚠️ Error al cargar variables: {str(e)}")
            return {"chat_history": []}

    def clear(self) -> None:
        """Limpia completamente la memoria"""
        self.memory.clear()
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
            logger.info("🧹 Memoria y archivo de persistencia limpiados")
        except Exception as e:
            logger.error(f"⚠️ Error al limpiar archivo: {str(e)}")