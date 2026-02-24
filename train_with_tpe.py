import optuna
import comet_ml
from train import TrainScript # Importing your existing logic

def objective(trial):
    # 1. Define the Search Space for weights
    # We use a range around your manual findings to see if better exists
    w_exp = trial.suggest_float("w_exp", 1.0, 10.0)
    w_col = trial.suggest_float("w_col", 1.0, 10.0)
    w_spa = trial.suggest_float("w_spa", 1.0, 10.0)
    w_tv  = trial.suggest_float("w_tv", 0.1, 5.0)
    w_glare = trial.suggest_float("w_glare", 1.0, 10.0)

    # 2. Initialize Comet Experiment for this specific trial
    experiment = comet_ml.start(project_name="AdaLOLIE-HPO-Study")
    experiment.log_parameters({
        "w_exp": w_exp, "w_col": w_col, "w_spa": w_spa, 
        "w_tv": w_tv, "w_glare": w_glare
    })
    
    weights = (w_exp, w_col, w_spa, w_tv, w_glare)

    # 3. Setup and Run a short training (e.g., 5-10 epochs)
    trainer = TrainScript(experiment)
    
    # Update the trainer's loss weights dynamically
    # Note: You'll need to update your AdaLOLIELoss forward pass to accept these
    trainer.custom_weights = (w_exp, w_col, w_spa, w_tv, w_glare)
    
    # Run training and get the best validation SSIM
    best_val_ssim = trainer.train_for_hpo(weights=weights, epochs=10) 
    
    experiment.end()
    
    # Return the metric Optuna should MAXIMIZE
    return best_val_ssim

if __name__ == "__main__":
    # Create a study to maximize SSIM
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20) # Run 20 different weight combinations

    print("Best weights found:")
    print(study.best_params)