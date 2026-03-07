import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path

# --- CONFIGURACIÓN ---
# Cargar variables de entorno
load_dotenv()

# Rutas (deben coincidir con las de create_embeddings.py)
SCRIPT_DIR = Path(__file__).resolve().parent
DB_DIRECTORY = SCRIPT_DIR / "chroma_db"
COLLECTION_NAME = "guia_embarazo_parto"

def test_database_query(query_text):
    """
    Carga la base de datos ChromaDB y realiza una consulta de prueba.
    """
    # Verificar si la base de datos existe
    if not DB_DIRECTORY.exists() or not any(DB_DIRECTORY.iterdir()):
        print(f"❌ Error: El directorio de la base de datos no existe o está vacío en {DB_DIRECTORY}")
        print("   Asegúrate de haber ejecutado 'create_embeddings.py' primero.")
        return

    print(f"✅ Directorio de la base de datos encontrado en: {DB_DIRECTORY}")

    # Inicializar el modelo de embeddings (debe ser el mismo que usaste para crear la BD)
    print("🧠 Inicializando modelo de embeddings de OpenAI...")
    try:
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    except Exception as e:
        print(f"❌ Error al inicializar el modelo de OpenAI: {e}")
        print("   Asegúrate de que tu OPENAI_API_KEY está configurada.")
        return

    # Cargar la base de datos vectorial existente
    print(f"💾 Cargando la base de datos vectorial desde: {DB_DIRECTORY}")
    try:
        db = Chroma(
            persist_directory=str(DB_DIRECTORY),
            embedding_function=embeddings_model,
            collection_name=COLLECTION_NAME
        )
        print(f"   Colección '{COLLECTION_NAME}' cargada con {db._collection.count()} vectores.")
    except Exception as e:
        print(f"❌ Error al cargar la base de datos Chroma: {e}")
        return

    # Realizar una búsqueda por similitud
    print(f"\n🔍 Realizando búsqueda por similitud para la consulta: '{query_text}'")
    try:
        # k=3 para obtener los 3 resultados más relevantes
        results = db.similarity_search(query_text, k=3)
        
        if not results:
            print("⚠️ La búsqueda no devolvió resultados.")
            return

        print("\n🎉 ¡Búsqueda completada! Estos son los 3 chunks más relevantes encontrados:\n")
        
        for i, doc in enumerate(results):
            print(f"--- RESULTADO {i+1} ---")
            print(f"📄 Contenido: \"{doc.page_content[:300]}...\"")
            print(f"   Metadata:")
            print(f"     - Fuente: {doc.metadata.get('source', 'N/A')}")
            print(f"     - Página: {doc.metadata.get('page_number', 'N/A')}")
            print(f"     - Sección: {doc.metadata.get('section_number', 'N/A')} - {doc.metadata.get('section_title', 'N/A')}")
            print("-" * 20 + "\n")

    except Exception as e:
        print(f"❌ Error durante la búsqueda por similitud: {e}")


def main():
    """Función principal para ejecutar la prueba."""
    print("=== INICIO DE LA PRUEBA DE RECUPERACIÓN ===")
    
    # Define una pregunta de prueba relevante para tus documentos
    sample_query = "¿Qué suplementos nutricionales se recomiendan en el embarazo?"
    
    test_database_query(sample_query)
    
    print("=== FIN DE LA PRUEBA ===")

if __name__ == "__main__":
    main()
