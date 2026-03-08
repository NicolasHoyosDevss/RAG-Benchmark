import json
import time
from pathlib import Path
from dotenv import load_dotenv
import os

from pageindex import PageIndexClient, PageIndexAPIError

# Carga variables de entorno desde .env (en la raiz del proyecto)
load_dotenv()

API_KEY = os.getenv("PAGEINDEX_API_KEY")
if not API_KEY:
    raise RuntimeError("Falta PAGEINDEX_API_KEY en .env")

client = PageIndexClient(api_key=API_KEY)

# PDF de entrada
pdf_path = Path("../Data/processed/guia_embarazo_parto_clean.pdf").resolve()
if not pdf_path.exists():
    raise FileNotFoundError(f"No existe: {pdf_path}")

try:
    # 1) Subir documento
    submit_resp = client.submit_document(file_path=str(pdf_path))
    doc_id = submit_resp["doc_id"]
    print("doc_id:", doc_id)

    # 2) Esperar hasta que esté listo para retrieval
    max_tries = 60
    for i in range(max_tries):
        tree_resp = client.get_tree(doc_id)
        ready = tree_resp.get("retrieval_ready", False)
        status = tree_resp.get("status", "unknown")
        print(f"[{i+1}/{max_tries}] status={status}, retrieval_ready={ready}")
        if ready:
            break
        time.sleep(2)
    else:
        raise TimeoutError("El documento no quedó listo para retrieval a tiempo")

    # 3) Lanzar una consulta de prueba
    query = "Resume este PDF en 5 puntos clave"
    retrieval_submit = client.submit_query(doc_id=doc_id, query=query)
    retrieval_id = retrieval_submit["retrieval_id"]
    print("retrieval_id:", retrieval_id)

    # 4) Polling del retrieval
    result = None
    for i in range(max_tries):
        retrieval_resp = client.get_retrieval(retrieval_id)
        r_status = retrieval_resp.get("status", "unknown")
        print(f"[retrieval {i+1}/{max_tries}] status={r_status}")

        if r_status in ("completed", "done", "success"):
            result = retrieval_resp
            break
        if r_status in ("failed", "error"):
            raise RuntimeError(f"Retrieval falló: {retrieval_resp}")

        time.sleep(2)

    if result is None:
        raise TimeoutError("No se obtuvo resultado de retrieval a tiempo")

    # 5) Guardar salida
    out_dir = Path("Data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"pageindex_result_{doc_id}.json"
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Resultado guardado en: {out_file}")

except PageIndexAPIError as e:
    print("Error de PageIndex API:", e)
    raise