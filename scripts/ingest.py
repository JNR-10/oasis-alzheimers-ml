import click
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import OASISDataLoader
from src.utils import save_json


@click.command()
@click.option('--cross-sectional', 
              type=click.Path(exists=True),
              default='oasis_cross-sectional-5708aa0a98d82080.xlsx',
              help='Path to OASIS cross-sectional dataset')
@click.option('--longitudinal',
              type=click.Path(exists=True),
              default='oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx',
              help='Path to OASIS longitudinal dataset')
@click.option('--output',
              type=click.Path(),
              default='data/raw/combined_oasis.csv',
              help='Output path for combined dataset')
def ingest(cross_sectional, longitudinal, output):
    click.echo("="*60)
    click.echo("OASIS Data Ingestion Pipeline")
    click.echo("="*60)
    
    loader = OASISDataLoader()
    
    click.echo(f"\n[1/5] Loading cross-sectional dataset from: {cross_sectional}")
    cs_df = loader.load_cross_sectional(cross_sectional)
    click.echo(f"  ✓ Loaded {cs_df.shape[0]} samples with {cs_df.shape[1]} features")
    
    click.echo(f"\n[2/5] Loading longitudinal dataset from: {longitudinal}")
    long_df = loader.load_longitudinal(longitudinal)
    click.echo(f"  ✓ Loaded {long_df.shape[0]} samples with {long_df.shape[1]} features")
    
    click.echo(f"\n[3/5] Analyzing datasets...")
    cs_info = loader.get_dataset_info(cs_df)
    long_info = loader.get_dataset_info(long_df)
    
    click.echo(f"\n  Cross-sectional dataset:")
    click.echo(f"    - Columns: {', '.join(cs_info['columns'][:5])}...")
    click.echo(f"    - Missing values: {sum(1 for v in cs_info['missing_values'].values() if v > 0)} columns with missing data")
    
    click.echo(f"\n  Longitudinal dataset:")
    click.echo(f"    - Columns: {', '.join(long_info['columns'][:5])}...")
    click.echo(f"    - Missing values: {sum(1 for v in long_info['missing_values'].values() if v > 0)} columns with missing data")
    
    click.echo(f"\n[4/5] Combining datasets...")
    combined_df, common_columns = loader.combine_datasets(cs_df, long_df)
    click.echo(f"  ✓ Combined dataset: {combined_df.shape[0]} samples with {combined_df.shape[1]} features")
    click.echo(f"  ✓ Common columns: {len(common_columns)}")
    
    click.echo(f"\n[5/5] Saving combined dataset to: {output}")
    loader.save_combined_data(combined_df, output)
    
    info_output = Path(output).parent / 'dataset_info.json'
    combined_info = loader.get_dataset_info(combined_df)
    save_json({
        'cross_sectional': cs_info,
        'longitudinal': long_info,
        'combined': combined_info,
        'common_columns': common_columns
    }, info_output)
    
    click.echo(f"\n{'='*60}")
    click.echo("✓ Data ingestion completed successfully!")
    click.echo(f"{'='*60}")
    click.echo(f"\nNext step: Run preprocessing")
    click.echo(f"  python scripts/preprocess.py")


if __name__ == '__main__':
    ingest()
