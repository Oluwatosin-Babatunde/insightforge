import sys
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

def textfile_to_pdf(input_txt_path: str, output_pdf_path: str):
    c = canvas.Canvas(output_pdf_path, pagesize=letter)
    width, height = letter

    x = 0.7 * inch
    y = height - 0.8 * inch

    c.setFont("Helvetica", 10)

    with open(input_txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n").replace("\t", "   ")

            # new page if needed
            if y < 0.8 * inch:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - 0.8 * inch

            # truncate very long lines safely
            c.drawString(x, y, line[:120])
            y -= 14

    c.save()

if __name__ == "__main__":
    input_txt = sys.argv[1]
    output_pdf = sys.argv[2]
    textfile_to_pdf(input_txt, output_pdf)
