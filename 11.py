from pathlib import Path

import pymupdf

project_dir = Path(__file__).resolve().parent
figure_dir = project_dir / "figures"

if not figure_dir.is_dir():
    raise FileNotFoundError(
        f"Figures directory does not exist: {figure_dir}"
    )

pdf_files = sorted(
    path
    for path in figure_dir.iterdir()
    if path.is_file() and path.suffix.lower() == ".pdf"
)

if not pdf_files:
    raise FileNotFoundError(
        f"No PDF files found in: {figure_dir}"
    )

print(f"Figures directory: {figure_dir}")
print(f"PDF files found: {len(pdf_files)}")

success_count = 0

for pdf_path in pdf_files:
    try:
        with pymupdf.open(pdf_path) as document:
            if len(document) != 1:
                print(
                    f"Skipped {pdf_path.name}: "
                    f"expected one page, found {len(document)}."
                )
                continue

            output_path = pdf_path.with_suffix(".png")

            document[0].get_pixmap(
                dpi=250,
                colorspace=pymupdf.csRGB,
                alpha=False,
            ).save(str(output_path))

        success_count += 1
        print(f"Created: {output_path.name}")

    except Exception as error:
        print(f"Failed: {pdf_path.name}: {error}")

print(f"Completed: {success_count}/{len(pdf_files)}")