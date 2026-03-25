import pandas as pd
import os
from pathlib import Path


class OASISDataLoader:
    def __init__(self, data_dir='data/raw'):
        self.data_dir = Path(data_dir)
        
    def load_cross_sectional(self, filepath):
        df = pd.read_excel(filepath)
        df['dataset'] = 'cross_sectional'
        if 'ID' in df.columns:
            df['Subject_ID'] = df['ID']
        return df
    
    def load_longitudinal(self, filepath):
        df = pd.read_excel(filepath)
        df['dataset'] = 'longitudinal'
        if 'Subject ID' in df.columns:
            df['Subject_ID'] = df['Subject ID']
        return df
    
    def combine_datasets(self, cross_sectional_df, longitudinal_df):
        common_columns = list(set(cross_sectional_df.columns) & set(longitudinal_df.columns))
        
        if 'Subject_ID' in cross_sectional_df.columns:
            common_columns.append('Subject_ID')
        
        cs_subset = cross_sectional_df[common_columns].copy()
        long_subset = longitudinal_df[common_columns].copy()
        
        combined_df = pd.concat([cs_subset, long_subset], axis=0, ignore_index=True)
        
        return combined_df, common_columns
    
    def get_dataset_info(self, df):
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
        }
        return info
    
    def save_combined_data(self, df, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Combined data saved to {output_path}")
