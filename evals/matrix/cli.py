"""Command-line interface for the Scenario Matrix generator & exporter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from evals.matrix.exporter import MatrixExporter
from evals.matrix.fixtures import get_default_matrix

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

console = Console(safe_box=True)


def display_matrix_table(matrix) -> None:
    """Print a Rich table of all scenarios in the matrix."""
    table = Table(title=f"{matrix.title} (v{matrix.version})", show_lines=True)
    table.add_column("ID", style="bold cyan", width=14)
    table.add_column("Category", style="magenta", width=12)
    table.add_column("Scenario Name", style="bold white", width=26)
    table.add_column("User Query", style="green", width=40)
    table.add_column("Expected Tools", style="yellow", width=18)
    table.add_column("Tags", style="dim", width=16)

    for s in matrix.scenarios:
        table.add_row(
            s.scenario_id,
            s.category.value,
            s.name,
            s.user_query,
            ", ".join(s.expected_tools) if s.expected_tools else "-",
            ", ".join(s.tags),
        )

    console.print(table)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for matrix generation and export."""
    parser = argparse.ArgumentParser(
        description="Scenario Matrix CLI: Export and inspect test scenario matrices."
    )
    parser.add_argument(
        "--export",
        "-e",
        type=str,
        help="Target file path to export (e.g. scenarios.xlsx, scenarios.csv, scenarios.html)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="evals/matrix_output",
        help="Output directory when exporting all formats (default: evals/matrix_output)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["xlsx", "csv", "html", "md", "all"],
        default="all",
        help="Export format (default: all)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all scenarios in console table",
    )

    args = parser.parse_args(argv)
    matrix = get_default_matrix()
    exporter = MatrixExporter(matrix)

    # 1. Print console table if requested or no export args
    if args.list or (not args.export and args.format == "all" and len(sys.argv) <= 1):
        display_matrix_table(matrix)

    # 2. Handle specific export target
    if args.export:
        target = Path(args.export)
        suffix = target.suffix.lower().lstrip(".")
        fmt = suffix if suffix in ("xlsx", "csv", "html", "md") else args.format

        if fmt == "xlsx":
            res = exporter.export_excel(target)
            console.print(f"[bold green][OK][/] Exported Excel matrix to: [bold]{res}[/]")
        elif fmt == "csv":
            res = exporter.export_csv(target)
            console.print(f"[bold green][OK][/] Exported CSV matrix to: [bold]{res}[/]")
        elif fmt == "html":
            res = exporter.export_html(target)
            console.print(f"[bold green][OK][/] Exported HTML matrix to: [bold]{res}[/]")
        elif fmt == "md":
            content = exporter.export_markdown()
            target.write_text(content, encoding="utf-8")
            console.print(f"[bold green][OK][/] Exported Markdown matrix to: [bold]{target}[/]")
        return 0

    # 3. Export all formats into output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    excel_file = out_dir / "scenarios.xlsx"
    csv_file = out_dir / "scenarios.csv"
    html_file = out_dir / "scenarios.html"
    md_file = out_dir / "scenarios.md"

    if args.format in ("xlsx", "all"):
        exporter.export_excel(excel_file)
        console.print(f"[bold green][OK][/] Excel:    [cyan]{excel_file}[/]")

    if args.format in ("csv", "all"):
        exporter.export_csv(csv_file)
        console.print(f"[bold green][OK][/] CSV:      [cyan]{csv_file}[/]")

    if args.format in ("html", "all"):
        exporter.export_html(html_file)
        console.print(f"[bold green][OK][/] HTML:     [cyan]{html_file}[/]")

    if args.format in ("md", "all"):
        md_file.write_text(exporter.export_markdown(), encoding="utf-8")
        console.print(f"[bold green][OK][/] Markdown: [cyan]{md_file}[/]")

    console.print(
        f"\n[bold green]Successfully generated Scenario Matrix across formats in:[/bold green] {out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
