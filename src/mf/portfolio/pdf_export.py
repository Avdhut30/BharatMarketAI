# src/mf/portfolio/pdf_export.py
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

def build_mf_summary_pdf(
    title: str,
    kpis: dict,
    table_rows: list,
    note: str = "Research-only. Not financial advice."
) -> bytes:
    """
    kpis: {"Invested": "₹1,00,000", "Current": "₹1,12,345", ...}
    table_rows: list of rows (list/tuple) like:
      [("Scheme", "Units", "Invested", "Current", "Gain", "CAGR"), (...)]
    """
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    x = 2 * cm
    y = height - 2 * cm

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title)
    y -= 1.0 * cm

    c.setFont("Helvetica", 10)
    for k, v in kpis.items():
        c.drawString(x, y, f"{k}: {v}")
        y -= 0.6 * cm

    y -= 0.2 * cm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x, y, "Scheme Summary (Top rows)")
    y -= 0.7 * cm

    c.setFont("Helvetica", 9)

    # Simple text table rendering
    max_rows_per_page = 28
    row_count = 0

    for row in table_rows:
        line = " | ".join(str(v) for v in row)
        c.drawString(x, y, line[:140])  # trim long lines
        y -= 0.5 * cm
        row_count += 1

        if row_count >= max_rows_per_page:
            c.showPage()
            c.setFont("Helvetica", 9)
            y = height - 2 * cm
            row_count = 0

    y -= 0.6 * cm
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(x, y, note)

    c.save()
    return buf.getvalue()
