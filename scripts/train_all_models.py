import click
import pandas as pd
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent))

from src.models import MLModel
from src.utils import save_json, print_metrics


@click.command()
@click.option('--data-dir',
              type=click.Path(exists=True),
              default='data/processed/oasis1',
              help='Directory containing processed data')
@click.option('--output-dir',
              type=click.Path(),
              default='models/phase1_oasis1',
              help='Output directory for trained models')
@click.option('--random-state',
              type=int,
              default=42,
              help='Random state for reproducibility')
def train_all(data_dir, output_dir, random_state):
    click.echo("="*70)
    click.echo("OASIS - Training All Models")
    click.echo("="*70)
    
    models_to_train = [
        'random_forest',
        'logistic_regression',
        'svm',
        'xgboost',
        'gradient_boosting',
        'knn',
        'naive_bayes',
        'adaboost'
    ]
    
    data_path = Path(data_dir)
    
    click.echo(f"\n[1/3] Loading processed data from: {data_dir}")
    X_train = pd.read_csv(data_path / 'X_train.csv')
    X_test = pd.read_csv(data_path / 'X_test.csv')
    y_train = pd.read_csv(data_path / 'y_train.csv')['target']
    y_test = pd.read_csv(data_path / 'y_test.csv')['target']
    
    click.echo(f"  ✓ Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    click.echo(f"  ✓ Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    click.echo(f"\n[2/3] Training {len(models_to_train)} models...")
    
    results = []
    
    for i, model_type in enumerate(models_to_train, 1):
        click.echo(f"\n{'='*70}")
        click.echo(f"Model {i}/{len(models_to_train)}: {model_type.upper()}")
        click.echo(f"{'='*70}")
        
        try:
            start_time = time.time()
            
            click.echo(f"  Initializing {model_type}...")
            ml_model = MLModel(model_type=model_type, random_state=random_state)
            
            click.echo(f"  Training...")
            ml_model.train(X_train, y_train)
            
            click.echo(f"  Evaluating...")
            metrics = ml_model.evaluate(X_test, y_test)
            
            training_time = time.time() - start_time
            metrics['training_time'] = training_time
            
            print_metrics(metrics, model_type)
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            model_file = output_path / f'{model_type}_model.pkl'
            ml_model.save_model(model_file)
            
            metrics_file = output_path / f'{model_type}_metrics.json'
            save_json(metrics, metrics_file)
            
            feature_importance = ml_model.get_feature_importance()
            if feature_importance is not None:
                importance_file = output_path / f'{model_type}_feature_importance.csv'
                feature_importance.to_csv(importance_file, index=False)
                click.echo(f"  ✓ Feature importance saved")
            
            results.append({
                'model': model_type,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc'],
                'training_time': training_time
            })
            
            click.echo(f"  ✓ {model_type} completed in {training_time:.2f}s")
            
        except Exception as e:
            click.echo(f"  ✗ Error training {model_type}: {str(e)}")
            results.append({
                'model': model_type,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'roc_auc': 0.0,
                'training_time': 0.0,
                'error': str(e)
            })
    
    click.echo(f"\n[3/3] Generating comparison report...")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    comparison_file = Path(output_dir) / 'model_comparison.csv'
    results_df.to_csv(comparison_file, index=False)
    
    click.echo(f"\n{'='*70}")
    click.echo("MODEL COMPARISON - Ranked by Accuracy")
    click.echo(f"{'='*70}")
    click.echo(f"\n{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC AUC':<10} {'Time(s)':<10}")
    click.echo("-"*70)
    
    for _, row in results_df.iterrows():
        click.echo(f"{row['model']:<20} {row['accuracy']:<10.4f} {row['precision']:<10.4f} {row['recall']:<10.4f} {row['f1_score']:<10.4f} {row['roc_auc']:<10.4f} {row['training_time']:<10.2f}")
    
    click.echo(f"\n{'='*70}")
    click.echo("✓ All models trained successfully!")
    click.echo(f"{'='*70}")
    click.echo(f"\nComparison saved to: {comparison_file}")
    click.echo(f"\nBest model: {results_df.iloc[0]['model']} (Accuracy: {results_df.iloc[0]['accuracy']:.4f})")
    click.echo(f"\nTo evaluate individual models:")
    click.echo(f"  python scripts/evaluate.py --model [model_name]")


if __name__ == '__main__':
    train_all()
