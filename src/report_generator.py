"""
report_generator.py
-------------------
Generates a professional PDF inspection report for LoomVision AI.

Uses ReportLab to create a multi-page document containing:
  1. Header with project branding, date, and session info
  2. Executive summary (total scans, defects, accuracy)
  3. Defect log table with timestamps, types, and image paths
  4. Embedded thumbnails of detected defect frames
  5. Footer with system information

Usage:
    from src.report_generator import generate_inspection_report
    pdf_path = generate_inspection_report(total_scans=1500)
"""

import os
import sqlite3
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, HRFlowable,
)


# ── Constants ────────────────────────────────────────────────────────
REPORT_DIR = "output/reports"
DB_PATH = "data/loomvision.db"
DEFECT_IMAGES_DIR = "output/defects"

# Brand colours
BRAND_PRIMARY = colors.HexColor("#38bdf8")      # sky-blue
BRAND_DARK = colors.HexColor("#0b0f19")          # dark bg
BRAND_ACCENT = colors.HexColor("#818cf8")        # indigo accent
BRAND_RED = colors.HexColor("#ef4444")            # defect red
BRAND_GREEN = colors.HexColor("#10b981")          # healthy green
BRAND_GREY = colors.HexColor("#94a3b8")           # muted text


def _ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def _get_defects_from_db(db_path: str, limit: int = 50) -> list[dict]:
    """Fetch recent defects from the SQLite database."""
    defects = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, timestamp, defect_type, image_path "
            "FROM defects ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        for row in cursor.fetchall():
            defects.append({
                "id": row[0],
                "timestamp": row[1],
                "defect_type": row[2],
                "image_path": row[3],
            })
        conn.close()
    except Exception as e:
        print(f"[Report] Warning: Could not read defects DB: {e}")
    return defects


def _get_defect_count(db_path: str) -> int:
    """Get total defect count from the database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM defects")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def _build_styles() -> dict:
    """Create custom paragraph styles for the report."""
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontSize=28,
            leading=34,
            textColor=BRAND_PRIMARY,
            spaceAfter=4 * mm,
            alignment=TA_LEFT,
            fontName="Helvetica-Bold",
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontSize=12,
            leading=16,
            textColor=BRAND_GREY,
            spaceAfter=8 * mm,
            alignment=TA_LEFT,
        ),
        "heading": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading2"],
            fontSize=16,
            leading=20,
            textColor=BRAND_PRIMARY,
            spaceBefore=10 * mm,
            spaceAfter=4 * mm,
            fontName="Helvetica-Bold",
        ),
        "body": ParagraphStyle(
            "BodyText",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#334155"),
        ),
        "metric_value": ParagraphStyle(
            "MetricValue",
            parent=base["Normal"],
            fontSize=22,
            leading=26,
            textColor=BRAND_PRIMARY,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            textColor=BRAND_GREY,
            alignment=TA_CENTER,
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=base["Normal"],
            fontSize=8,
            leading=10,
            textColor=BRAND_GREY,
            alignment=TA_CENTER,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.white,
            fontName="Helvetica-Bold",
        ),
        "table_cell": ParagraphStyle(
            "TableCell",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#1e293b"),
        ),
    }
    return styles


def generate_inspection_report(
    total_scans: int = 0,
    db_path: str = DB_PATH,
    output_dir: str = REPORT_DIR,
) -> str:
    """
    Generate a professional PDF inspection report.

    Parameters
    ----------
    total_scans : int
        Total number of frames scanned in the session.
    db_path : str
        Path to the SQLite defects database.
    output_dir : str
        Directory where the PDF will be saved.

    Returns
    -------
    str
        Absolute path to the generated PDF file.
    """
    _ensure_dir(output_dir)

    timestamp = datetime.now()
    filename = f"LoomVision_Report_{timestamp.strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(output_dir, filename)

    styles = _build_styles()
    defects = _get_defects_from_db(db_path, limit=50)
    total_defects = _get_defect_count(db_path)

    if total_scans <= 0:
        total_scans = max(total_defects * 10, 1)  # rough estimate fallback

    accuracy = 100.0 - ((total_defects / max(total_scans, 1)) * 100.0)

    # ── Build document ────────────────────────────────────────────
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title="LoomVision AI — Inspection Report",
        author="LoomVision AI",
    )

    story = []

    # ────────────────────────────────────────────────────────────
    # 1. HEADER
    # ────────────────────────────────────────────────────────────
    story.append(Paragraph("LoomVision AI", styles["title"]))
    story.append(Paragraph(
        f"Fabric Defect Inspection Report &nbsp;•&nbsp; "
        f"{timestamp.strftime('%B %d, %Y at %I:%M %p')}",
        styles["subtitle"],
    ))

    story.append(HRFlowable(
        width="100%", thickness=1, color=BRAND_PRIMARY,
        spaceAfter=6 * mm, spaceBefore=2 * mm,
    ))

    # ────────────────────────────────────────────────────────────
    # 2. EXECUTIVE SUMMARY — KPI Cards
    # ────────────────────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", styles["heading"]))

    kpi_data = [
        [
            Paragraph(f"{total_scans:,}", styles["metric_value"]),
            Paragraph(f"{total_defects}", styles["metric_value"]),
            Paragraph(f"{accuracy:.1f}%", styles["metric_value"]),
        ],
        [
            Paragraph("Total Scans", styles["metric_label"]),
            Paragraph("Defects Found", styles["metric_label"]),
            Paragraph("Accuracy Rate", styles["metric_label"]),
        ],
    ]

    kpi_table = Table(kpi_data, colWidths=[55 * mm, 55 * mm, 55 * mm])
    kpi_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (0, -1), 0.5, BRAND_PRIMARY),
        ("BOX", (1, 0), (1, -1), 0.5, BRAND_PRIMARY),
        ("BOX", (2, 0), (2, -1), 0.5, BRAND_PRIMARY),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("TOPPADDING", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 10),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 6 * mm))

    # Status paragraph
    if total_defects == 0:
        status_text = (
            '<font color="#10b981"><b>✓ No defects detected.</b></font> '
            "The fabric passed all quality checks during this inspection session."
        )
    else:
        status_text = (
            f'<font color="#ef4444"><b>⚠ {total_defects} defect(s) detected.</b></font> '
            "Review the defect log below for details. "
            "Affected fabric sections should be inspected manually."
        )
    story.append(Paragraph(status_text, styles["body"]))

    # ────────────────────────────────────────────────────────────
    # 3. DEFECT LOG TABLE
    # ────────────────────────────────────────────────────────────
    if defects:
        story.append(Paragraph("Defect Log", styles["heading"]))
        story.append(Paragraph(
            f"Showing the {len(defects)} most recent defects "
            f"(out of {total_defects} total).",
            styles["body"],
        ))
        story.append(Spacer(1, 3 * mm))

        # Table header
        header_row = [
            Paragraph("ID", styles["table_header"]),
            Paragraph("Timestamp", styles["table_header"]),
            Paragraph("Defect Type", styles["table_header"]),
        ]

        table_data = [header_row]
        for d in defects:
            table_data.append([
                Paragraph(str(d["id"]), styles["table_cell"]),
                Paragraph(d["timestamp"], styles["table_cell"]),
                Paragraph(d["defect_type"] or "Unknown", styles["table_cell"]),
            ])

        defect_table = Table(
            table_data,
            colWidths=[20 * mm, 50 * mm, 95 * mm],
            repeatRows=1,
        )
        defect_table.setStyle(TableStyle([
            # Header row
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            # Body rows
            ("BACKGROUND", (0, 1), (-1, -1), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                colors.white, colors.HexColor("#f8fafc")
            ]),
            # Grid
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
            ("BOX", (0, 0), (-1, -1), 0.6, BRAND_PRIMARY),
            # Padding
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(defect_table)

        # ────────────────────────────────────────────────────────
        # 4. DEFECT IMAGE GALLERY (up to 6 most recent)
        # ────────────────────────────────────────────────────────
        images_to_show = []
        for d in defects[:6]:
            img_path = d.get("image_path", "")
            if img_path and os.path.exists(img_path):
                images_to_show.append((d, img_path))

        if images_to_show:
            story.append(Paragraph("Defect Image Gallery", styles["heading"]))
            story.append(Paragraph(
                "Captured frames at the moment of defect detection.",
                styles["body"],
            ))
            story.append(Spacer(1, 3 * mm))

            for defect_info, img_path in images_to_show:
                try:
                    img = RLImage(img_path, width=140 * mm, height=100 * mm)
                    img.hAlign = "CENTER"
                    story.append(img)
                    story.append(Paragraph(
                        f'<font color="#64748b" size="8">'
                        f'ID #{defect_info["id"]} — {defect_info["defect_type"]} '
                        f'— {defect_info["timestamp"]}'
                        f'</font>',
                        ParagraphStyle(
                            "ImgCaption", alignment=TA_CENTER,
                            fontSize=8, textColor=BRAND_GREY,
                            spaceBefore=2 * mm, spaceAfter=6 * mm,
                        ),
                    ))
                except Exception as e:
                    print(f"[Report] Could not embed image {img_path}: {e}")

    # ────────────────────────────────────────────────────────────
    # 5. FOOTER / SYSTEM INFO
    # ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 10 * mm))
    story.append(HRFlowable(
        width="100%", thickness=0.5, color=BRAND_GREY,
        spaceAfter=4 * mm, spaceBefore=4 * mm,
    ))
    story.append(Paragraph(
        f"Generated by LoomVision AI &nbsp;•&nbsp; "
        f"LoomVision AI Project &nbsp;•&nbsp; "
        f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} &nbsp;•&nbsp; "
        f"Engine: Dynamic PatchCore (ResNet-18)",
        styles["footer"],
    ))

    # ── Build PDF ─────────────────────────────────────────────
    doc.build(story)
    abs_path = os.path.abspath(pdf_path)
    print(f"[Report] ✅ PDF report generated: {abs_path}")
    return abs_path


# ── Quick test ────────────────────────────────────────────────────
if __name__ == "__main__":
    path = generate_inspection_report(total_scans=1500)
    print(f"Report saved to: {path}")
