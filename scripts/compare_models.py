import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


@click.command()
@click.option('--comparison-file',
              type=click.Path(exists=True),
              default='models/model_comparison.csv',
              help='Path to model comparison CSV')
@click.option('--output-dir',
              type=click.Path(),
              default='results',
              help='Output directory for visualizations')
def compare(comparison_file, output_dir):
    click.echo("="*70)
    click.echo("Model Comparison Visualization")
    click.echo("="*70)
    
    df = pd.read_csv(comparison_file)
    df = df.sort_values('accuracy', ascending=False)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    click.echo(f"\n[1/4] Creating accuracy comparison plot...")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='model', y='accuracy', palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0.75, 1.0)
    for i, v in enumerate(df['accuracy']):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    click.echo(f"  ✓ Saved to {output_path / 'accuracy_comparison.png'}")
    
    click.echo(f"\n[2/4] Creating ROC AUC comparison plot...")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='model', y='roc_auc', palette='coolwarm')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('ROC AUC')
    plt.xlabel('Model')
    plt.title('Model ROC AUC Comparison')
    plt.ylim(0.85, 1.0)
    for i, v in enumerate(df['roc_auc']):
        plt.text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path / 'roc_auc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    click.echo(f"  ✓ Saved to {output_path / 'roc_auc_comparison.png'}")
    
    click.echo(f"\n[3/4] Creating precision-recall comparison plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for idx, row in df.iterrows():
        ax.scatter(row['recall'], row['precision'], s=200, alpha=0.6, label=row['model'])
        ax.annotate(row['model'], (row['recall'], row['precision']), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision vs Recall Trade-off', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(0.6, 1.0)
    plt.tight_layout()
    plt.savefig(output_path / 'precision_recall_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    click.echo(f"  ✓ Saved to {output_path / 'precision_recall_comparison.png'}")
    
    click.echo(f"\n[4/4] Creating comprehensive metrics heatmap...")
    metrics_df = df[['model', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].set_index('model')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics_df.T, annot=True, fmt='.4f', cmap='YlGnBu', 
                cbar_kws={'label': 'Score'}, vmin=0.6, vmax=1.0)
    plt.title('Model Performance Heatmap')
    plt.xlabel('Model')
    plt.ylabel('Metric')
    plt.tight_layout()
    plt.savefig(output_path / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    click.echo(f"  ✓ Saved to {output_path / 'metrics_heatmap.png'}")
    
    click.echo(f"\n{'='*70}")
    click.echo("✓ Comparison visualizations created successfully!")
    click.echo(f"{'='*70}")
    click.echo(f"\nGenerated files:")
    click.echo(f"  - {output_path / 'accuracy_comparison.png'}")
    click.echo(f"  - {output_path / 'roc_auc_comparison.png'}")
    click.echo(f"  - {output_path / 'precision_recall_comparison.png'}")
    click.echo(f"  - {output_path / 'metrics_heatmap.png'}")
    
    click.echo(f"\n{'='*70}")
    click.echo("Top 3 Models:")
    click.echo(f"{'='*70}")
    for i, row in df.head(3).iterrows():
        click.echo(f"\n{i+1}. {row['model'].upper()}")
        click.echo(f"   Accuracy:  {row['accuracy']:.4f}")
        click.echo(f"   Precision: {row['precision']:.4f}")
        click.echo(f"   Recall:    {row['recall']:.4f}")
        click.echo(f"   F1 Score:  {row['f1_score']:.4f}")
        click.echo(f"   ROC AUC:   {row['roc_auc']:.4f}")


if __name__ == '__main__':
    compare()
