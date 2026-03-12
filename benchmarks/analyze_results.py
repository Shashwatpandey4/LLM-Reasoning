import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_jsonl(filepath: str) -> pd.DataFrame:
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Flatten the metrics dictionary
    df = pd.json_normalize(data)
    return df

def analyze_results(filepath: str, output_dir: str):
    print(f"Analyzing {filepath}...")
    df = load_jsonl(filepath)
    
    # Needs ground truth and predictions to compute accuracy per sample
    # Re-calculate correct boolean flag for plotting
    def check_correct(row):
        pred = str(row['extracted_answer']).strip() if pd.notna(row['extracted_answer']) else ""
        truth = str(row['ground_truth']).strip() if pd.notna(row['ground_truth']) else ""
        try:
             return abs(float(pred) - float(truth)) < 1e-5
        except:
             return pred == truth

    df['is_correct'] = df.apply(check_correct, axis=1)
    
    # Set seaborn styling for professional papers
    sns.set_theme(style="whitegrid", context="talk")
    palette = {"Correct": "#2ecc71", "Incorrect": "#e74c3c"}
    df['Status'] = df['is_correct'].map({True: 'Correct', False: 'Incorrect'})
    
    base_name = os.path.basename(filepath).replace('.jsonl', '')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Boxplot + Swarmplot: Distribution of Reasoning Tokens
    if 'metrics.reasoning_tokens' in df.columns:
        plt.figure(figsize=(12, 7))
        sns.boxplot(data=df, x='Status', y='metrics.reasoning_tokens', palette=palette, showfliers=False, order=["Correct", "Incorrect"])
        sns.stripplot(data=df, x='Status', y='metrics.reasoning_tokens', color='black', alpha=0.3, jitter=True, order=["Correct", "Incorrect"])
        plt.title('Reasoning Effort vs. Accuracy (Gemma-3-1b-it on GSM8K)')
        plt.xlabel('Result')
        plt.ylabel('Reasoning Tokens Used')
        
        # Add summary stats to plot
        corr_mean = df[df['is_correct']]['metrics.reasoning_tokens'].mean()
        incorr_mean = df[~df['is_correct']]['metrics.reasoning_tokens'].mean()
        if pd.notna(corr_mean) and pd.notna(incorr_mean):
           plt.text(0, df['metrics.reasoning_tokens'].max() * 0.95, f"Mean: {corr_mean:.1f}", ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
           plt.text(1, df['metrics.reasoning_tokens'].max() * 0.95, f"Mean: {incorr_mean:.1f}", ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        plt.tight_layout()
        plt_path = os.path.join(output_dir, f"{base_name}_reasoning_boxplot.png")
        plt.savefig(plt_path, dpi=300)
        plt.close()
        print(f"Saved {plt_path}")

    # 2. Rolling Accuracy by Reasoning Length (Sorting by Length to see if longer reasoning = better outcome)
    if 'metrics.reasoning_tokens' in df.columns:
        df_sorted = df.sort_values(by='metrics.reasoning_tokens').reset_index(drop=True)
        # Calculate moving average of accuracy
        window_size = max(5, len(df) // 10) # 10% window
        df_sorted['rolling_accuracy'] = df_sorted['is_correct'].astype(float).rolling(window=window_size, center=True).mean() * 100
        
        plt.figure(figsize=(12, 7))
        # Plot individual points (0 or 100) thinly
        plt.scatter(df_sorted['metrics.reasoning_tokens'], df_sorted['is_correct'].astype(float)*100, alpha=0.1, color='grey', label='Raw (0 or 1)')
        # Plot the smoothed trend
        sns.lineplot(data=df_sorted, x='metrics.reasoning_tokens', y='rolling_accuracy', color="#3498db", linewidth=3, label=f'Rolling Accuracy (window={window_size})')
        
        plt.title('Does longer reasoning lead to better accuracy?')
        plt.xlabel('Reasoning Tokens Used')
        plt.ylabel('Exact Match Accuracy (%)')
        plt.ylim(-5, 105)
        plt.legend()
        plt.tight_layout()
        plt_path = os.path.join(output_dir, f"{base_name}_reasoning_vs_accuracy_curve.png")
        plt.savefig(plt_path, dpi=300)
        plt.close()
        print(f"Saved {plt_path}")
        
    print("Analysis Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze reasoning results")
    parser.add_argument("input_file", help="Path to the JSONL results file")
    parser.add_argument("--output_dir", default="results/plots", help="Directory to save plots")
    args = parser.parse_args()
    
    analyze_results(args.input_file, args.output_dir)
