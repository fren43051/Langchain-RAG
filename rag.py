import os
import sys
import warnings
import time
import urllib.request
import json

# 1. Silenciamos las advertencias de deprecación y de sistema
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="No languages specified")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# --- CONFIGURACIÓN PARA LANGSMITH ---
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "Mi-Proyecto-RAG"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def comprobar_ollama(model_embed, model_llm):
    print("=" * 60)
    print("🔍 DIAGNÓSTICO DE OLLAMA")
    print("-" * 60)
    print("-> Verificando conexión con Ollama en http://localhost:11434...")
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            modelos_locales = [m["name"] for m in data.get("models", [])]
            print(f"   [OK] Ollama está ejecutándose correctamente.")
            print(f"   Modelos disponibles localmente: {modelos_locales}")
            
            # Comprobar embedding
            embed_ok = False
            for m in modelos_locales:
                if model_embed in m:
                    embed_ok = True
                    model_embed_actual = m
                    break
            
            if not embed_ok:
                print(f"   ⚠️ [ALERTA] El modelo de embeddings '{model_embed}' no parece estar descargado.")
                print(f"            Ejecuta en tu terminal: ollama pull {model_embed}")
            else:
                print(f"   [OK] Modelo de embeddings '{model_embed}' disponible ({model_embed_actual}).")
                
            # Comprobar LLM
            llm_ok = False
            for m in modelos_locales:
                if model_llm in m:
                    llm_ok = True
                    model_llm_actual = m
                    break
            
            if not llm_ok:
                print(f"   ⚠️ [ALERTA] El modelo LLM '{model_llm}' no parece estar descargado.")
                print(f"            Ejecuta en tu terminal: ollama pull {model_llm}")
            else:
                print(f"   [OK] Modelo LLM '{model_llm}' disponible ({model_llm_actual}).")
    except Exception as e:
        print("   ❌ [ERROR] No se pudo conectar con Ollama.")
        print("              Asegúrate de que Ollama esté ejecutándose en tu sistema.")
        print("              Puedes iniciarlo abriendo la aplicación Ollama en tu computadora.")
        print(f"              Detalles del error: {e}")
    print("=" * 60 + "\n")

# Comprobaciones previas de Ollama
comprobar_ollama("bge-m3", "gemma3:4b")

# Importaciones de Langchain
from langchain_community.document_loaders import DirectoryLoader
from transformers import AutoTokenizer
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser

# 2. Cargamos los PDFs especificando el idioma
print("-> 1. Cargando PDFs desde el directorio 'documentos'...")
t0 = time.time()
pdfs = []
try:
    # Intentamos primero con PyPDFDirectoryLoader que es mucho más rápido y liviano
    print("   Intentando usar PyPDFDirectoryLoader...")
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    loader = PyPDFDirectoryLoader('documentos')
    pdfs = loader.load()
    print(f"   [OK] PyPDFDirectoryLoader cargó {len(pdfs)} páginas de documentos.")
except Exception as e:
    print(f"   PyPDFDirectoryLoader no disponible o falló: {e}")
    print("   Usando DirectoryLoader por defecto (puede ser lento o colgarse con unstructured)...")
    try:
        loader = DirectoryLoader(
            'documentos',
            glob='*.pdf',
            loader_kwargs={"languages": ["spa"]}
        )
        pdfs = loader.load()
        print(f"   [OK] DirectoryLoader cargó {len(pdfs)} documentos.")
    except Exception as ex:
        print(f"   ❌ Error crítico al cargar documentos: {ex}")
        sys.exit(1)
print(f"   Tiempo transcurrido en carga de documentos: {time.time() - t0:.2f} segundos\n")

# Configurar Tokenizer y Splitter
print("-> 2. Configurando Tokenizer y División de Texto...")
t0 = time.time()
try:
    print("   Intentando descargar/cargar tokenizer 'BAAI/bge-m3' desde Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3', local_files_only=False)
    splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=1250,
        chunk_overlap=150
    )
    fragmentos = splitter.split_documents(pdfs)
    print(f"   [OK] Documentos divididos usando el tokenizer BAAI/bge-m3.")
except Exception as e:
    print(f"   ⚠️ No se pudo cargar el tokenizer de Hugging Face (puede ser por falta de conexión): {e}")
    print("   Usando divisor de texto alternativo local (RecursiveCharacterTextSplitter) sin descargas...")
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    fragmentos = splitter.split_documents(pdfs)
    print(f"   [OK] Documentos divididos usando RecursiveCharacterTextSplitter local.")

print(f"   Total de fragmentos generados: {len(fragmentos)}")
print(f"   Tiempo transcurrido en división de texto: {time.time() - t0:.2f} segundos\n")

# Configurar embeddings de Ollama
print("-> 3. Inicializando embeddings con Ollama (modelo: bge-m3)...")
t0 = time.time()
embeddings = OllamaEmbeddings(model='bge-m3')
print(f"   Embeddings configurados.")

# Configurar vectorstore y retriever
print("-> 4. Creando base de datos vectorial FAISS (generando embeddings para cada fragmento)...")
print("   (Aviso: Si tienes muchos fragmentos o estás en CPU, esto puede tomar de 30 segundos a unos minutos)")
t0_embed = time.time()
try:
    vector_store = FAISS.from_documents(fragmentos, embeddings)
    retriever = vector_store.as_retriever()
    print(f"   [OK] Base de datos FAISS creada exitosamente en {time.time() - t0_embed:.2f} segundos.")
except Exception as e:
    print(f"   ❌ [ERROR CRÍTICO] Error al generar embeddings con Ollama: {e}")
    print("      ¿Ollama se detuvo? ¿Descargaste el modelo con 'ollama pull bge-m3'?")
    sys.exit(1)
print(f"   Tiempo total de indexación: {time.time() - t0:.2f} segundos\n")

# 3. Configurar el Prompt
# 3. Configurar Prompts y Modelos para el Pipeline

# Reescritor de preguntas
query_model = OllamaLLM(model="gemma3:1b")
rewriter_prompt_template = """
Genera la consulta de búsqueda para la base de datos de vectores (Vector DB) a partir de una pregunta del usuario,
permitiendo una respuesta más precisa por medio de la búsqueda semántica.
Basta devolver la consulta revisada del Vector DB, entre comillas.

# PREGUNTA DEL USUARIO: {user_question}
# CONSULTA REVISADA DEL VECTOR DB:
"""
rewriter_prompt = PromptTemplate.from_template(rewriter_prompt_template)
rewriter_chain = rewriter_prompt | query_model | StrOutputParser()

# Prompt para el RAG final
prompt = ChatPromptTemplate.from_messages([
    ("system" , "Responde usando exclusivamente el contenido que se incluye a continuación: \n\n {contexto}"),
    ("human", "{query}")
])
modelo = OllamaLLM(model = "gemma3:4b")

# 4. Ejecución del RAG con Pipeline Completo
pregunta = 'Cómo solicitar el seguro de viaje?'
#
# rag_chain = (
#     {"contexto": RunnablePassthrough() | rewriter_chain | retriever,
#      "query": RunnablePassthrough()}
#     | prompt | modelo | StrOutputParser()
# )
#
# respuesta = rag_chain.invoke(pregunta)
# print(respuesta)

template_multipregunta = """
Eres un asistente de inteligencia artificial experto en optimización de consultas.
Tu tarea es generar exactamente cinco versiones diferentes de la pregunta del usuario para buscar documentos en una base de datos vectorial.
Al generar múltiples perspectivas, ayudas a superar las limitaciones de la búsqueda semántica basada en distancia.

Debes cumplir estrictamente con las siguientes reglas:
1. Devuelve únicamente las 5 preguntas alternativas.
2. Devuélvelas en una sola línea separadas únicamente por comas.
3. NO agregues números, guiones, viñetas ni texto adicional de introducción o de cierre.

# PREGUNTA ORIGINAL: {question}

# PREGUNTAS ALTERNATIVAS (separadas por comas):
"""

prompt_multipregunta = PromptTemplate.from_template(template_multipregunta)
chain_multipregunta = prompt_multipregunta | modelo | CommaSeparatedListOutputParser()

# Invocamos la cadena (que devuelve automáticamente una lista de Python)
preguntas = chain_multipregunta.invoke({"question": pregunta})

print(preguntas)

rag_chain = (
    {"contexto": RunnablePassthrough() | retriever,
     "query": RunnablePassthrough()}
    | prompt | modelo | StrOutputParser()
)

for p in preguntas:
    print(f"\n🔍 Ejecutando RAG para: {p}")
    intentos = 3
    for intento in range(intentos):
        try:
            respuesta = rag_chain.invoke(p)
            print(respuesta)
            break
        except Exception as e:
            if intento < intentos - 1:
                print(f"   ⚠️ Conexión con Ollama interrumpida. Reintentando en 3 segundos... (Intento {intento + 2}/{intentos})")
                time.sleep(3)
            else:
                print(f"   ❌ Fallaron todos los intentos al conectar con Ollama: {e}")
    time.sleep(1)


