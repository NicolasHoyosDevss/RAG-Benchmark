import os
import time
from pageindex import PageIndexClient, PageIndexAPIError

API_KEY = os.getenv("PAGEINDEX_API_KEY")
if not API_KEY:
    raise RuntimeError("Falta PAGEINDEX_API_KEY")

client = PageIndexClient(api_key=API_KEY)

try:
    # 1) Subir PDF
    submit = client.submit_document(file_path="ruta/a/tu_documento.pdf")
    doc_id = submit["doc_id"]
    print("doc_id:", doc_id)

    # 2) Esperar a que esté listo para retrieval
    for _ in range(30):
        tree = client.get_tree(doc_id)
        ready = tree.get("retrieval_ready", False)
        status = tree.get("status")
        print("status:", status, "retrieval_ready:", ready)
        if ready:
            break
        time.sleep(2)
    else:
        raise TimeoutError("El documento no quedó listo a tiempo")

    # 3) Hacer query de retrieval
    q = client.submit_query(doc_id=doc_id, query="Resume los puntos clave del documento")
    retrieval_id = q["retrieval_id"]
    print("retrieval_id:", retrieval_id)

    # 4) Consultar resultado de retrieval
    for _ in range(30):
        r = client.get_retrieval(retrieval_id)
        r_status = r.get("status")
        print("retrieval status:", r_status)
        if r_status in ("completed", "done", "success"):
            print("resultado:", r)
            break
        if r_status in ("failed", "error"):
            raise RuntimeError(f"Retrieval falló: {r}")
        time.sleep(2)

    # 5) Chat completions (opcional, usando el doc_id)
    chat = client.chat_completions(
        messages=[{"role": "user", "content": "Dame un resumen ejecutivo en 5 bullets"}],
        doc_id=doc_id,
        stream=False,
    )
    print("chat:", chat)

except PageIndexAPIError as e:
    print("Error API:", e)