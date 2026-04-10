"""CLI for datamix."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional


def _build_cli():  # type: ignore[no-untyped-def]
    try:
        import click
    except ImportError:
        raise SystemExit("CLI dependencies required: pip install datamix[cli]")

    @click.group()
    @click.version_option(package_name="datamix")
    def cli() -> None:
        """datamix — dataset mixing & curriculum optimizer."""

    @cli.command()
    @click.argument("jsonl_path", type=click.Path(exists=True))
    @click.option("--text-key", default="text", help="JSON key for text field.")
    @click.option("--max-examples", type=int, default=None)
    def profile(jsonl_path: str, text_key: str, max_examples: Optional[int]) -> None:
        """Profile a JSONL dataset file."""
        from datamix.profile import profile_jsonl

        p = profile_jsonl(jsonl_path, text_key=text_key, max_examples=max_examples)

        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table

            console = Console()
            header = Table(show_header=False, box=None, padding=(0, 2))
            header.add_column(style="bold cyan", width=22)
            header.add_column()
            header.add_row("Dataset", p.name)
            header.add_row("Examples", f"{p.n_examples:,}")
            header.add_row("Tokens (est.)", f"{p.n_tokens:,}")
            header.add_row("Avg tokens/example", f"{p.avg_tokens_per_example:.1f}")
            header.add_row("Avg length (chars)", f"{p.quality.avg_length:.1f}")
            header.add_row("Min / Max length", f"{p.quality.min_length:,} / {p.quality.max_length:,}")
            console.print(Panel(header, title="[bold]Dataset Profile[/bold]", border_style="blue"))
        except ImportError:
            click.echo(json.dumps(p.to_dict(), indent=2))

    @cli.command()
    @click.argument("jsonl_paths", nargs=-1, type=click.Path(exists=True))
    @click.option("--strategy", type=click.Choice(["proportional", "equal", "temperature"]), default="proportional")
    @click.option("--budget", type=int, default=0, help="Total token budget.")
    def mix(jsonl_paths: tuple, strategy: str, budget: int) -> None:
        """Create a mix recipe from multiple JSONL files."""
        from datamix.mixer import create_recipe
        from datamix._types import MixStrategy
        from datamix.profile import profile_jsonl

        profiles = [profile_jsonl(p) for p in jsonl_paths]
        strat = MixStrategy(strategy)
        recipe = create_recipe(profiles, strategy=strat, total_tokens=budget)
        click.echo(json.dumps(recipe.to_dict(), indent=2))

    @cli.command()
    @click.argument("jsonl_path", type=click.Path(exists=True))
    @click.option("--min-length", type=int, default=0)
    @click.option("--max-length", type=int, default=0)
    @click.option("--dedup/--no-dedup", default=True)
    def clean(jsonl_path: str, min_length: int, max_length: int, dedup: bool) -> None:
        """Clean a JSONL file — filter and deduplicate."""
        from datamix.quality import dedup_exact, length_filter

        examples = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "") if isinstance(obj, dict) else str(obj)
                    if text:
                        examples.append(text)
                except json.JSONDecodeError:
                    continue

        click.echo(f"Loaded {len(examples):,} examples")

        if min_length > 0 or max_length > 0:
            examples, stats = length_filter(examples, min_length=min_length, max_length=max_length)
            click.echo(f"Length filter: kept {stats['kept']:,}, removed {stats['removed']:,}")

        if dedup:
            examples, stats = dedup_exact(examples)
            click.echo(f"Dedup: kept {stats['kept']:,}, removed {stats['duplicates_removed']:,}")

        click.echo(f"Final: {len(examples):,} examples")

    return cli


cli = _build_cli()
