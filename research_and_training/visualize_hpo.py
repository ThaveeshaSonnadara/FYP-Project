import optuna
import pandas as pd
import plotly.express as px
import plotly.io as pio
import os

def generate_parallel_coordinates_plot(study_name="adalolie_hpo_study", storage="sqlite:///hpo.db"):
    """
    Generates a Parallel Coordinates Plot from an Optuna study.
    This helps visualize the trade-offs between different loss weights and the resulting metrics,
    now including No-Reference metrics (NIQE, BRISQUE).
    """
    # 1. Load the study
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        df = study.trials_dataframe()
        
        # Clean up column names (Optuna prefix: 'params_', 'user_attrs_', 'value')
        df.columns = [c.replace('params_', '').replace('user_attrs_', '') for c in df.columns]
        
    except Exception as e:
        print(f"⚠️ Error loading Optuna study: {e}")
        print("Generating sample plot with extended metric placeholder data...")
        # Extended dummy data with No-Reference metrics included
        data = {
            'w_exp': [8, 5, 3, 7, 5, 4],
            'w_col': [0.5, 5, 2, 8, 4, 6],
            'w_spa': [5, 5, 7, 10, 8, 6],
            'w_tv': [2, 0.5, 1, 0.2, 0.5, 1.5],
            'w_glare': [10, 5, 3, 5, 7, 4],
            'PSNR': [17.1, 22.1, 19.5, 21.2, 22.5, 20.8],
            'SSIM': [0.81, 0.89, 0.85, 0.87, 0.90, 0.86],
            'NIQE': [8.5, 4.2, 5.8, 4.9, 4.1, 5.1],       # Lower is better
            'BRISQUE': [45.2, 28.5, 36.1, 31.0, 27.9, 32.4] # Lower is better
        }
        df = pd.DataFrame(data)

    # 2. Select columns to visualize
    # These are the hyperparameters and target metrics
    cols = ['w_exp', 'w_col', 'w_spa', 'w_tv', 'w_glare', 'PSNR', 'SSIM', 'NIQE', 'BRISQUE']
    
    # Ensure all columns exist in the dataframe
    available_cols = [c for c in cols if c in df.columns]
    
    if 'NIQE' in available_cols:
        # We color the graph by NIQE since for Unsupervised models, it's a stronger indicator.
        # Note: For NIQE/BRISQUE, lower is better, so we reverse the color scale.
        color_col = "NIQE"
        c_scale = px.colors.diverging.Tealrose_r 
    else:
        color_col = "SSIM"
        c_scale = px.colors.diverging.Tealrose

    # 3. Create the Plotly Parallel Coordinates Plot
    fig = px.parallel_coordinates(
        df, 
        dimensions=available_cols,
        color=color_col, 
        color_continuous_scale=c_scale,
        title="Parallel Coordinates: Loss Weights vs. Multi-Metric Performance"
    )

    # 4. Save the plot
    output_dir = "../Output/Safety_Report"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save as HTML (Interactive - best for presentation)
    html_path = os.path.join(output_dir, "hpo_parallel_coordinates.html")
    fig.write_html(html_path)
    
    # Save as PNG (Static - best for the document/thesis)
    png_path = os.path.join(output_dir, "hpo_parallel_coordinates.png")
    fig.write_image(png_path)
    
    # Export the trial data to CSV for the appendix
    csv_path = os.path.join(output_dir, "hpo_trial_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"✅ HPO Visualization Complete!")
    print(f"   - Interactive Plot: {html_path}")
    print(f"   - Static Plot: {png_path}")
    print(f"   - Data Export: {csv_path}")

if __name__ == "__main__":
    generate_parallel_coordinates_plot()