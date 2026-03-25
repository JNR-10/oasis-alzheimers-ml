import click
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


@click.command()
@click.option('--cross-sectional', 
              type=click.Path(exists=True),
              default='oasis_cross-sectional-5708aa0a98d82080.xlsx',
              help='Path to OASIS cross-sectional dataset')
@click.option('--longitudinal',
              type=click.Path(exists=True),
              default='oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx',
              help='Path to OASIS longitudinal dataset')
def explore(cross_sectional, longitudinal):
    click.echo("="*70)
    click.echo("OASIS Dataset Exploration")
    click.echo("="*70)
    
    click.echo(f"\n{'='*70}")
    click.echo("CROSS-SECTIONAL DATASET")
    click.echo("="*70)
    
    cs_df = pd.read_excel(cross_sectional)
    
    click.echo(f"\nShape: {cs_df.shape[0]} rows × {cs_df.shape[1]} columns")
    
    click.echo(f"\nColumns:")
    for i, col in enumerate(cs_df.columns, 1):
        click.echo(f"  {i:2d}. {col}")
    
    click.echo(f"\nData Types:")
    for col, dtype in cs_df.dtypes.items():
        click.echo(f"  {col:30s} {dtype}")
    
    click.echo(f"\nMissing Values:")
    missing = cs_df.isnull().sum()
    missing_pct = (missing / len(cs_df) * 100)
    for col in cs_df.columns:
        if missing[col] > 0:
            click.echo(f"  {col:30s} {missing[col]:4d} ({missing_pct[col]:5.1f}%)")
    
    click.echo(f"\nFirst 5 rows:")
    click.echo(cs_df.head().to_string())
    
    click.echo(f"\nSummary Statistics:")
    click.echo(cs_df.describe().to_string())
    
    click.echo(f"\n{'='*70}")
    click.echo("LONGITUDINAL DATASET")
    click.echo("="*70)
    
    long_df = pd.read_excel(longitudinal)
    
    click.echo(f"\nShape: {long_df.shape[0]} rows × {long_df.shape[1]} columns")
    
    click.echo(f"\nColumns:")
    for i, col in enumerate(long_df.columns, 1):
        click.echo(f"  {i:2d}. {col}")
    
    click.echo(f"\nData Types:")
    for col, dtype in long_df.dtypes.items():
        click.echo(f"  {col:30s} {dtype}")
    
    click.echo(f"\nMissing Values:")
    missing = long_df.isnull().sum()
    missing_pct = (missing / len(long_df) * 100)
    for col in long_df.columns:
        if missing[col] > 0:
            click.echo(f"  {col:30s} {missing[col]:4d} ({missing_pct[col]:5.1f}%)")
    
    click.echo(f"\nFirst 5 rows:")
    click.echo(long_df.head().to_string())
    
    click.echo(f"\nSummary Statistics:")
    click.echo(long_df.describe().to_string())
    
    click.echo(f"\n{'='*70}")
    click.echo("DATASET COMPARISON")
    click.echo("="*70)
    
    common_cols = set(cs_df.columns) & set(long_df.columns)
    cs_only = set(cs_df.columns) - set(long_df.columns)
    long_only = set(long_df.columns) - set(cs_df.columns)
    
    click.echo(f"\nCommon columns ({len(common_cols)}):")
    for col in sorted(common_cols):
        click.echo(f"  - {col}")
    
    if cs_only:
        click.echo(f"\nCross-sectional only ({len(cs_only)}):")
        for col in sorted(cs_only):
            click.echo(f"  - {col}")
    
    if long_only:
        click.echo(f"\nLongitudinal only ({len(long_only)}):")
        for col in sorted(long_only):
            click.echo(f"  - {col}")
    
    click.echo(f"\n{'='*70}")
    click.echo("✓ Exploration completed!")
    click.echo("="*70)


if __name__ == '__main__':
    explore()
