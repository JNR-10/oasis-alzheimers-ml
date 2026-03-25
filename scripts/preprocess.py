import click
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessor import OASISPreprocessor
from src.utils import save_json


@click.command()
@click.option('--input',
              type=click.Path(exists=True),
              default='data/raw/combined_oasis.csv',
              help='Path to combined dataset')
@click.option('--output-dir',
              type=click.Path(),
              default='data/processed',
              help='Output directory for processed data')
@click.option('--test-size',
              type=float,
              default=0.2,
              help='Test set size (default: 0.2)')
@click.option('--random-state',
              type=int,
              default=42,
              help='Random state for reproducibility')
def preprocess(input, output_dir, test_size, random_state):
    click.echo("="*60)
    click.echo("OASIS Data Preprocessing Pipeline")
    click.echo("="*60)
    
    click.echo(f"\n[1/7] Loading combined dataset from: {input}")
    df = pd.read_csv(input)
    click.echo(f"  ✓ Loaded {df.shape[0]} samples with {df.shape[1]} features")
    
    preprocessor = OASISPreprocessor()
    
    click.echo(f"\n[2/7] Identifying target and features...")
    target_col, feature_cols = preprocessor.identify_target_and_features(df)
    click.echo(f"  ✓ Target column: {target_col}")
    click.echo(f"  ✓ Feature columns: {len(feature_cols)}")
    click.echo(f"    Features: {', '.join(feature_cols[:5])}...")
    
    click.echo(f"\n[3/7] Analyzing missing values...")
    missing_before = df.isnull().sum().sum()
    click.echo(f"  ✓ Total missing values: {missing_before}")
    
    click.echo(f"\n[4/7] Running preprocessing pipeline...")
    click.echo(f"  - Handling missing values")
    click.echo(f"  - Encoding categorical variables")
    click.echo(f"  - Creating binary target")
    click.echo(f"  - Splitting data (test_size={test_size}, subject-level split)")
    click.echo(f"  - Scaling features")
    
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        df, target_col, feature_cols, test_size=test_size, random_state=random_state, subject_level_split=True
    )
    
    click.echo(f"\n[5/7] Preprocessing results:")
    click.echo(f"  ✓ Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    click.echo(f"  ✓ Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    click.echo(f"  ✓ Class distribution (train): {y_train.value_counts().to_dict()}")
    click.echo(f"  ✓ Class distribution (test): {y_test.value_counts().to_dict()}")
    
    click.echo(f"\n[6/7] Saving processed data to: {output_dir}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    X_train.to_csv(output_path / 'X_train.csv', index=False)
    X_test.to_csv(output_path / 'X_test.csv', index=False)
    y_train.to_csv(output_path / 'y_train.csv', index=False, header=['target'])
    y_test.to_csv(output_path / 'y_test.csv', index=False, header=['target'])
    
    click.echo(f"  ✓ X_train.csv")
    click.echo(f"  ✓ X_test.csv")
    click.echo(f"  ✓ y_train.csv")
    click.echo(f"  ✓ y_test.csv")
    
    click.echo(f"\n[7/7] Saving preprocessor...")
    preprocessor.save_preprocessor(output_path)
    
    metadata = {
        'target_column': target_col,
        'feature_columns': feature_cols,
        'final_features': preprocessor.feature_names,
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'num_features': int(X_train.shape[1]),
        'test_size': test_size,
        'random_state': random_state,
        'class_distribution_train': y_train.value_counts().to_dict(),
        'class_distribution_test': y_test.value_counts().to_dict()
    }
    save_json(metadata, output_path / 'preprocessing_metadata.json')
    
    click.echo(f"\n{'='*60}")
    click.echo("✓ Data preprocessing completed successfully!")
    click.echo(f"{'='*60}")
    click.echo(f"\nNext step: Run model training")
    click.echo(f"  python scripts/train.py --model random_forest")
    click.echo(f"  python scripts/train.py --model logistic_regression")


if __name__ == '__main__':
    preprocess()
