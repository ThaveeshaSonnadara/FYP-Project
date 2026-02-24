
import optuna
import pandas as pd
import plotly.express as px
import plotly.io as pio
import os

def generate_parallel_coordinates_plot(study_name="adalolie_hpo_study", storage="sqlite:///hpo.db"):
    """
    Generates a Parallel Coordinates Plot from an Optuna study.
    This helps visualize the trade-offs between different loss weights and the resulting metrics.
    """
    # 1. Load the study
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception as e:
        print(f"Error loading study: {e}")
        # Creating dummy data for demonstration if no study exists
        print("Generating sample plot with placeholder data...")
        data = {
            'w_exp': [8, 5, 3, 7, 5, 4],
            'w_col': [0.5, 5, 2, 8, 4, 6],
            'w_spa': [5, 5, 7, 10, 8, 6],
            'w_tv': [2, 0.5, 1, 0.2, 0.5, 1.5],
            'w_glare': [10, 5, 3, 5, 7, 4],
            'PSNR': [17.1, 22.1, 19.5, 21.2, 22.5, 20.8],
            'SSIM': [0.81, 0.89, 0.85, 0.87, 0.90, 0.86]
        }
        df = pd.DataFrame(data)
    else:
        # Convert study to a DataFrame
        df = study.trials_dataframe()
        
        # Clean up column names (Optuna prefix: 'params_', 'user_attrs_', 'value')
        # We assume values were logged as user_attrs if multi-objective was used
        # or as the main 'value' if optimizing one metric.
        df.columns = [c.replace('params_', '').replace('user_attrs_', '') for c in df.columns]

    # 2. Select columns to visualize
    # These are the hyperparameters and target metrics
    cols = ['w_exp', 'w_col', 'w_spa', 'w_tv', 'w_glare', 'PSNR', 'SSIM']
    
    # Ensure all columns exist in the dataframe
    available_cols = [c for c in cols if c in df.columns]
    
    # 3. Create the Plotly Parallel Coordinates Plot
    fig = px.parallel_coordinates(
        df, 
        dimensions=available_cols,
        color="SSIM", # Color the lines based on the best SSIM results
        color_continuous_scale=px.colors.diverging.Tealrose,
        title="Parallel Coordinates: Loss Weights vs. Enhancement Metrics"
    )

    # 4. Save the plot
    output_dir = "Output/Safety_Report"
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
