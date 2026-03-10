import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pageindex import PageIndexAPIError, PageIndexClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga el árbol de un documento en PageIndex y lo guarda como JSON local."
    )

    parser.add_argument("doc_id", help="ID del documento en PageIndex (ejemplo: pi-xxxx)")
    parser.add_argument(
        "--node-summary",
        action="store_true",
        help="Si se incluye, solicita resumen por nodo en la respuesta del árbol.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Carpeta de salida. Por defecto usa data/processed en la raíz del proyecto.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env", override=True)

    api_key = os.getenv("PAGEINDEX_API_KEY")
    if not api_key:
        raise RuntimeError("Falta PAGEINDEX_API_KEY en el archivo .env")

    output_dir = args.output_dir if args.output_dir else (project_root / "data" / "processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    client = PageIndexClient(api_key=api_key)

    try:
        tree = client.get_tree(args.doc_id, node_summary=args.node_summary)
    except PageIndexAPIError as exc:
        print(f"Error de PageIndex API al traer árbol: {exc}")
        raise

    out_file = output_dir / f"pageindex_tree_{args.doc_id}.json"
    out_file.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Árbol guardado correctamente:")
    print(out_file)


if __name__ == "__main__":
    main()
