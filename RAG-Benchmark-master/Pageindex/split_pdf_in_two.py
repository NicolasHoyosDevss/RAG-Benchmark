from pathlib import Path
import argparse

from PyPDF2 import PdfReader, PdfWriter


def split_pdf_in_two(input_pdf: Path, output_dir: Path) -> tuple[Path, Path]:
    reader = PdfReader(str(input_pdf))
    total_pages = len(reader.pages)

    if total_pages < 2:
        raise ValueError("El PDF debe tener al menos 2 paginas para dividirlo en dos.")

    # Si el total es impar, la primera mitad queda con una pagina extra.
    split_index = (total_pages + 1) // 2

    writer_1 = PdfWriter()
    writer_2 = PdfWriter()

    for i in range(split_index):
        writer_1.add_page(reader.pages[i])

    for i in range(split_index, total_pages):
        writer_2.add_page(reader.pages[i])

    output_dir.mkdir(parents=True, exist_ok=True)

    part_1 = output_dir / f"{input_pdf.stem}_parte_1.pdf"
    part_2 = output_dir / f"{input_pdf.stem}_parte_2.pdf"

    with part_1.open("wb") as f1:
        writer_1.write(f1)

    with part_2.open("wb") as f2:
        writer_2.write(f2)

    return part_1, part_2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Divide un PDF en dos mitades y guarda dos archivos PDF."
    )
    parser.add_argument(
        "input_pdf",
        type=Path,
        help="Ruta al PDF de entrada. Ejemplo: ../Data/raw/archivo.pdf",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../Data/processed"),
        help="Carpeta de salida para los PDFs generados (por defecto: ../Data/processed).",
    )

    args = parser.parse_args()

    input_pdf = args.input_pdf.resolve()
    output_dir = args.output_dir.resolve()

    if not input_pdf.exists():
        raise FileNotFoundError(f"No existe el archivo: {input_pdf}")

    if input_pdf.suffix.lower() != ".pdf":
        raise ValueError("El archivo de entrada debe ser un PDF (.pdf).")

    part_1, part_2 = split_pdf_in_two(input_pdf, output_dir)

    print("PDF dividido correctamente:")
    print(f"- {part_1}")
    print(f"- {part_2}")


if __name__ == "__main__":
    main()
