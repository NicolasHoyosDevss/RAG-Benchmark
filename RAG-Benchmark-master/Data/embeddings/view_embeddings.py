import chromadb
from pathlib import Path
import json

# --- CONFIGURACIÓN ---
SCRIPT_DIR = Path(__file__).resolve().parent
DB_DIRECTORY = SCRIPT_DIR / "chroma_db"
COLLECTION_NAME = "guia_embarazo_parto"


def view_collection_data(db_directory, collection_name):
    """
    Carga una colección de ChromaDB y muestra su contenido.
    """
    print(f"🔍 Cargando base de datos desde: {db_directory}")
    print(f"   Colección: {collection_name}")

    try:
        # Conectar con la base de datos persistente
        client = chromadb.PersistentClient(path=str(db_directory))

        # Obtener la colección
        collection = client.get_collection(name=collection_name)

        # Obtener el número total de items
        count = collection.count()
        print(
            f"✅ Colección '{collection_name}' cargada. Contiene {count} items.")

        if count == 0:
            print("   La colección está vacía.")
            return

        # Recuperar todos los datos de la colección
        # Esto puede consumir mucha memoria si la colección es muy grande
        print("\n📦 Recuperando todos los datos de la colección (documentos, metadatos y embeddings)...")
        data = collection.get(
            include=['metadatas', 'documents', 'embeddings']
        )

        # Imprimir la información de cada item
        for i in range(len(data['ids'])):
            print("\n" + "="*50)
            print(f"  ID: {data['ids'][i]}")
            print(
                f"  Metadata: {json.dumps(data['metadatas'][i], indent=2, ensure_ascii=False)}")
            # Muestra los primeros 200 caracteres
            print(f"  Documento: {data['documents'][i][:200]}...")

            # Muestra la forma y los primeros elementos del embedding
            embedding = data['embeddings'][i]
            print(f"  Embedding (vector):")
            print(f"    - Dimensiones: {len(embedding)}")
            print(f"    - Primeros 5 valores: {embedding[:5]}")
            print("="*50)

        print(
            f"\n✅ Se han mostrado los {len(data['ids'])} items de la colección.")

    except Exception as e:
        print(f"❌ Error al cargar o leer la colección: {e}")
        print("   Asegúrate de que la ruta a la base de datos y el nombre de la colección son correctos.")
        print("   Y que el script `create_embeddings.py` se ha ejecutado correctamente.")


def main():
    """Función principal."""
    print("=== VISUALIZADOR DE DATOS DE CHROMADB ===")
    view_collection_data(DB_DIRECTORY, COLLECTION_NAME)
    print("\n=== FIN DEL PROCESO ===")


if __name__ == "__main__":
    main()
