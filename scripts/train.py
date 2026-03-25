import click
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models import MLModel
from src.utils import save_json, print_metrics


@click.command()
@click.option('--model',
              type=click.Choice(['random_forest', 'logistic_regression', 'svm', 'xgboost', 'gradient_boosting', 'knn', 'naive_bayes', 'adaboost']),
              required=True,
              help='Model type to train')
@click.option('--data-dir',
              type=click.Path(exists=True),
              default='data/processed',
              help='Directory containing processed data')
@click.option('--output-dir',
              type=click.Path(),
              default='models',
              help='Output directory for trained models')
@click.option('--random-state',
              type=int,
              default=42,
              help='Random state for reproducibility')
def train(model, data_dir, output_dir, random_state):
    click.echo("="*60)
    click.echo(f"OASIS Model Training Pipeline - {model.upper()}")
    click.echo("="*60)
    
    data_path = Path(data_dir)
    
    click.echo(f"\n[1/5] Loading processed data from: {data_dir}")
    X_train = pd.read_csv(data_path / 'X_train.csv')
    X_test = pd.read_csv(data_path / 'X_test.csv')
    y_train = pd.read_csv(data_path / 'y_train.csv')['target']
    y_test = pd.read_csv(data_path / 'y_test.csv')['target']
    
    click.echo(f"  ✓ Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    click.echo(f"  ✓ Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    click.echo(f"\n[2/5] Initializing {model} model...")
    ml_model = MLModel(model_type=model, random_state=random_state)
    click.echo(f"  ✓ Model initialized")
    
    click.echo(f"\n[3/5] Training model...")
    ml_model.train(X_train, y_train)
    click.echo(f"  ✓ Training completed")
    
    click.echo(f"\n[4/5] Evaluating model on test set...")
    metrics = ml_model.evaluate(X_test, y_test)
    print_metrics(metrics, model)
    
    click.echo(f"\n[5/5] Saving model and results...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_file = output_path / f'{model}_model.pkl'
    ml_model.save_model(model_file)
    
    metrics_file = output_path / f'{model}_metrics.json'
    save_json(metrics, metrics_file)
    
    feature_importance = ml_model.get_feature_importance()
    if feature_importance is not None:
        importance_file = output_path / f'{model}_feature_importance.csv'
        feature_importance.to_csv(importance_file, index=False)
        click.echo(f"  ✓ Feature importance saved to {importance_file}")
        
        click.echo(f"\n  Top 10 Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            if model == 'random_forest':
                click.echo(f"    {idx+1}. {row['feature']}: {row['importance']:.4f}")
            else:
                click.echo(f"    {idx+1}. {row['feature']}: {row['coefficient']:.4f}")
    
    click.echo(f"\n{'='*60}")
    click.echo("✓ Model training completed successfully!")
    click.echo(f"{'='*60}")
    click.echo(f"\nNext step: Run model evaluation")
    click.echo(f"  python scripts/evaluate.py --model {model}")


if __name__ == '__main__':
    train()
