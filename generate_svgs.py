"""Generate SVG terminal screenshots for datamix README."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns

# ── SVG 1: Mix Recipe + Budget ──

console = Console(record=True, width=90)

header = Table(show_header=False, box=None, padding=(0, 2))
header.add_column(style="bold cyan", width=22)
header.add_column()
header.add_row("Recipe", "mix_temperature")
header.add_row("Strategy", "temperature (T=1.5)")
header.add_row("Components", "4 datasets")
header.add_row("Total tokens", "2,000,000,000")
console.print(Panel(header, title="[bold]datamix — Mix Recipe[/bold]", border_style="blue"))

table = Table(title="Dataset Weights & Token Allocation")
table.add_column("Dataset", style="bold")
table.add_column("Weight", justify="right")
table.add_column("Tokens", justify="right")
table.add_column("Source Size", justify="right")
table.add_column("Epochs", justify="right")
table.add_column("Bar", min_width=20)

data = [
    ("wikipedia", "0.312", "624M", "1.2B", "0.52", 31),
    ("code-python", "0.278", "556M", "800M", "0.70", 28),
    ("arxiv-papers", "0.245", "490M", "600M", "0.82", 25),
    ("books-fiction", "0.165", "330M", "400M", "0.83", 17),
]
for name, weight, tokens, source, epochs, bar_len in data:
    bar = "[green]" + "█" * bar_len + "░" * (31 - bar_len) + "[/green]"
    table.add_row(name, weight, tokens, source, epochs, bar)

console.print(table)
console.print()

budget = Table(title="Token Budget Report", show_lines=True)
budget.add_column("Metric", style="bold")
budget.add_column("Value", justify="right")
budget.add_row("Total budget", "[bold]2,000,000,000[/bold] tokens")
budget.add_row("Allocated", "2,000,000,000 tokens")
budget.add_row("Overflow", "[green]0[/green] tokens")
budget.add_row("Utilization", "[green]100.0%[/green]")
console.print(budget)

svg = console.export_svg(title="datamix recipe")
with open("/data/users/zacharie/repogen/datamix/assets/recipe.svg", "w") as f:
    f.write(svg)
print(f"recipe.svg: {len(svg):,} bytes")

# ── SVG 2: Curriculum Schedule ──

console2 = Console(record=True, width=90)

sched = Table(title="Curriculum Schedule — Cosine Decay", show_lines=True)
sched.add_column("Phase", style="bold")
sched.add_column("Progress")
sched.add_column("wikipedia", justify="right")
sched.add_column("code", justify="right")
sched.add_column("arxiv", justify="right")
sched.add_column("books", justify="right")

sched.add_row("warmup", "0% → 25%", "[bold cyan]0.73[/bold cyan]", "0.09", "0.09", "0.09")
sched.add_row("ramp", "25% → 50%", "[cyan]0.50[/cyan]", "0.17", "0.17", "0.17")
sched.add_row("main", "50% → 75%", "[yellow]0.27[/yellow]", "0.24", "0.24", "0.24")
sched.add_row("cooldown", "75% → 100%", "[red]0.07[/red]", "[bold green]0.31[/bold green]", "[bold green]0.31[/bold green]", "[bold green]0.31[/bold green]")

console2.print(Panel(sched, title="[bold]datamix — Curriculum[/bold]", border_style="blue"))
console2.print()

quality = Table(title="Quality Pipeline")
quality.add_column("Step", style="bold")
quality.add_column("Input", justify="right")
quality.add_column("Output", justify="right")
quality.add_column("Removed", justify="right")

quality.add_row("Length filter (50-10000)", "1,250,000", "1,198,000", "[yellow]52,000 (4.2%)[/yellow]")
quality.add_row("Exact dedup", "1,198,000", "1,082,000", "[yellow]116,000 (9.7%)[/yellow]")
quality.add_row("Near-dedup (5-gram)", "1,082,000", "1,024,000", "[yellow]58,000 (5.4%)[/yellow]")
quality.add_row("Quality score > 0.5", "1,024,000", "[bold green]956,000[/bold green]", "[yellow]68,000 (6.6%)[/yellow]")

console2.print(quality)

svg2 = console2.export_svg(title="datamix curriculum")
with open("/data/users/zacharie/repogen/datamix/assets/curriculum.svg", "w") as f:
    f.write(svg2)
print(f"curriculum.svg: {len(svg2):,} bytes")
