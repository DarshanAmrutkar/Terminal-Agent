"""Multi-format exporter for Scenario Matrix (Excel .xlsx, CSV, HTML, Markdown)."""

from __future__ import annotations

import csv
from pathlib import Path

from evals.matrix.models import ScenarioMatrix

# Category color accents for Excel badges & HTML
CATEGORY_COLORS: dict[str, dict[str, str]] = {
    "repo_qa": {"bg": "DBEAFE", "fg": "1E40AF", "html": "#2563EB"},
    "security": {"bg": "FEE2E2", "fg": "991B1B", "html": "#DC2626"},
    "bug_fix": {"bg": "FEF3C7", "fg": "92400E", "html": "#D97706"},
    "refactor": {"bg": "F3E8FF", "fg": "6B21A8", "html": "#7C3AED"},
    "performance": {"bg": "D1FAE5", "fg": "065F46", "html": "#059669"},
    "persistence": {"bg": "E0E7FF", "fg": "3730A3", "html": "#4F46E5"},
    "cli": {"bg": "F1F5F9", "fg": "334155", "html": "#475569"},
}


class MatrixExporter:
    """Exports a ScenarioMatrix to formatted Excel, CSV, HTML, and Markdown files."""

    def __init__(self, matrix: ScenarioMatrix) -> None:
        self.matrix = matrix

    def export_excel(self, output_path: str | Path) -> Path:
        """Export matrix to a professionally styled Excel (.xlsx) workbook."""
        try:
            import openpyxl
            from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
            from openpyxl.utils import get_column_letter
        except ImportError as err:
            raise ImportError(
                "openpyxl is required for Excel export. Install it with `pip install openpyxl`."
            ) from err

        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        wb = openpyxl.Workbook()

        # -------------------------------------------------------------
        # Sheet 1: Scenario Matrix Table
        # -------------------------------------------------------------
        ws = wb.active
        ws.title = "Scenario Matrix"
        ws.views.sheetView[0].showGridLines = True

        headers = [
            "Scenario ID",
            "Name",
            "Category",
            "User Query",
            "Preconditions",
            "Expected Tools",
            "Forbidden Tools",
            "Expected Concepts",
            "Forbidden Patterns",
            "Allowed Permissions",
            "Typical Response",
            "Pass Criteria",
            "Max Turns",
            "Tags",
        ]

        # Header styling
        header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        header_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

        thin_side = Side(border_style="thin", color="CBD5E1")
        cell_border = Border(top=thin_side, left=thin_side, right=thin_side, bottom=thin_side)

        ws.append(headers)
        ws.row_dimensions[1].height = 28

        for col_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = header_align
            cell.border = cell_border

        # Body Rows
        alt_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
        white_fill = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")

        data_rows = [s.to_row_dict() for s in self.matrix.scenarios]

        for row_idx, data in enumerate(data_rows, start=2):
            row_values = [data[h] for h in headers]
            ws.append(row_values)
            ws.row_dimensions[row_idx].height = 60  # generous height for multiline text

            is_alt = (row_idx % 2 == 0)
            base_fill = alt_fill if is_alt else white_fill

            for col_idx, col_name in enumerate(headers, start=1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.border = cell_border
                cell.fill = base_fill
                cell.font = Font(name="Calibri", size=10)

                # Column-specific alignments
                if col_name in ("Scenario ID", "Category", "Max Turns"):
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)

                # Color-badge Category cells
                if col_name == "Category":
                    cat = str(cell.value)
                    if cat in CATEGORY_COLORS:
                        colors = CATEGORY_COLORS[cat]
                        cell.fill = PatternFill(
                            start_color=colors["bg"], end_color=colors["bg"], fill_type="solid"
                        )
                        cell.font = Font(name="Calibri", size=10, bold=True, color=colors["fg"])

        # Auto-fit column widths with sensible bounds
        col_width_defaults = {
            "Scenario ID": 16,
            "Name": 26,
            "Category": 16,
            "User Query": 38,
            "Preconditions": 28,
            "Expected Tools": 22,
            "Forbidden Tools": 22,
            "Expected Concepts": 30,
            "Forbidden Patterns": 26,
            "Allowed Permissions": 22,
            "Typical Response": 44,
            "Pass Criteria": 30,
            "Max Turns": 12,
            "Tags": 20,
        }
        for col_idx, col_name in enumerate(headers, start=1):
            col_letter = get_column_letter(col_idx)
            ws.column_dimensions[col_letter].width = col_width_defaults.get(col_name, 20)

        # -------------------------------------------------------------
        # Sheet 2: Matrix Overview & Statistics
        # -------------------------------------------------------------
        ws_summary = wb.create_sheet(title="Matrix Overview")
        ws_summary.views.sheetView[0].showGridLines = True

        ws_summary.append([self.matrix.title])
        ws_summary.cell(row=1, column=1).font = Font(name="Calibri", size=16, bold=True, color="1E293B")
        ws_summary.append([f"Version: {self.matrix.version} | Total Scenarios: {len(self.matrix.scenarios)}"])
        ws_summary.append([])

        # Category Breakdown
        ws_summary.append(["Category Breakdown", "Count"])
        ws_summary.cell(row=4, column=1).font = Font(bold=True)
        ws_summary.cell(row=4, column=2).font = Font(bold=True)

        cat_counts: dict[str, int] = {}
        for s in self.matrix.scenarios:
            cat_counts[s.category.value] = cat_counts.get(s.category.value, 0) + 1

        for cat, count in sorted(cat_counts.items()):
            ws_summary.append([cat, count])

        ws_summary.column_dimensions["A"].width = 30
        ws_summary.column_dimensions["B"].width = 15

        wb.save(target)
        return target

    def export_csv(self, output_path: str | Path) -> Path:
        """Export matrix to an RFC 4180 standard CSV file."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        rows = [s.to_row_dict() for s in self.matrix.scenarios]
        if not rows:
            target.write_text("", encoding="utf-8")
            return target

        fieldnames = list(rows[0].keys())
        with open(target, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return target

    def export_html(self, output_path: str | Path) -> Path:
        """Export matrix to an interactive, responsive standalone HTML document."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        rows = [s.to_row_dict() for s in self.matrix.scenarios]
        headers = list(rows[0].keys()) if rows else []

        # Generate HTML table rows
        table_rows: list[str] = []
        for r in rows:
            cat = r.get("Category", "")
            cat_badge_color = CATEGORY_COLORS.get(cat, {}).get("html", "#475569")

            cells: list[str] = []
            for h in headers:
                val = str(r[h])
                if h == "Category":
                    cell_html = f'<span class="badge" style="background-color: {cat_badge_color}20; color: {cat_badge_color}; border: 1px solid {cat_badge_color}50;">{val}</span>'
                elif h == "Scenario ID":
                    cell_html = f'<strong><code>{val}</code></strong>'
                elif h in ("Expected Tools", "Forbidden Tools", "Expected Concepts"):
                    items = [f'<code>{it.strip()}</code>' for it in val.split(",") if it.strip()]
                    cell_html = " ".join(items) if items else "(none)"
                else:
                    cell_html = val.replace("\n", "<br>")
                cells.append(f"<td>{cell_html}</td>")
            table_rows.append("<tr>" + "".join(cells) + "</tr>")

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{self.matrix.title}</title>
  <style>
    :root {{
      --bg: #F8FAFC;
      --surface: #FFFFFF;
      --border: #E2E8F0;
      --text: #0F172A;
      --text-muted: #64748B;
      --primary: #2563EB;
      --header-bg: #1E293B;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      margin-bottom: 24px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
    }}
    h1 {{
      font-size: 24px;
      margin: 0 0 8px 0;
      color: var(--header-bg);
    }}
    .meta {{
      color: var(--text-muted);
      font-size: 14px;
    }}
    .search-box {{
      margin-bottom: 16px;
      display: flex;
      gap: 12px;
      align-items: center;
    }}
    .search-box input {{
      padding: 8px 14px;
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 14px;
      width: 320px;
    }}
    .table-container {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow-x: auto;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }}
    th {{
      background: var(--header-bg);
      color: #FFFFFF;
      padding: 12px 14px;
      font-weight: 600;
      white-space: nowrap;
    }}
    td {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
      max-width: 320px;
      word-wrap: break-word;
    }}
    tr:nth-child(even) td {{
      background-color: #F8FAFC;
    }}
    tr:hover td {{
      background-color: #F1F5F9;
    }}
    code {{
      background: #F1F5F9;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 11px;
      font-family: Consolas, monospace;
      color: #0F172A;
    }}
    .badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 9999px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{self.matrix.title}</h1>
    <div class="meta">Version {self.matrix.version} &bull; Total Scenarios: {len(self.matrix.scenarios)}</div>
  </header>

  <div class="search-box">
    <input type="text" id="searchInput" placeholder="Filter scenarios by keyword, category, tool..." onkeyup="filterTable()">
  </div>

  <div class="table-container">
    <table id="matrixTable">
      <thead>
        <tr>
          {"".join(f"<th>{h}</th>" for h in headers)}
        </tr>
      </thead>
      <tbody>
        {"".join(table_rows)}
      </tbody>
    </table>
  </div>

  <script>
    function filterTable() {{
      const query = document.getElementById('searchInput').value.toLowerCase();
      const rows = document.querySelectorAll('#matrixTable tbody tr');
      rows.forEach(row => {{
        const text = row.innerText.toLowerCase();
        row.style.display = text.includes(query) ? '' : 'none';
      }});
    }}
  </script>
</body>
</html>
"""
        target.write_text(html_content, encoding="utf-8")
        return target

    def export_markdown(self) -> str:
        """Generate a GitHub Flavored Markdown summary table."""
        rows = [s.to_row_dict() for s in self.matrix.scenarios]
        if not rows:
            return "*(No scenarios in matrix)*"

        cols = ["Scenario ID", "Name", "Category", "User Query", "Expected Tools", "Forbidden Tools"]
        lines = [
            f"# {self.matrix.title}",
            "",
            f"**Total Scenarios:** {len(self.matrix.scenarios)} | **Version:** {self.matrix.version}",
            "",
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join([":---"] * len(cols)) + " |",
        ]
        for r in rows:
            line_vals = [str(r[c]).replace("\n", " ") for c in cols]
            lines.append("| " + " | ".join(line_vals) + " |")

        return "\n".join(lines)
