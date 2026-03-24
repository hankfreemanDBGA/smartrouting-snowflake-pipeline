These scripts run from the repo root (snowflake_full_pipeline/) with cwd set by the runner.

- run_train_22k_validate_march.py — 22k splits + 5-tower train + validatemarch + comprehensive comparison
- run_retrain_all_22k_validate_march.py — full retrain including CatBoost replica, KS sweep, best config, 4-tower/3-tower, eval

Other scripts here are invoked by those runners or run manually; REPO_ROOT is the parent folder (snowflake_full_pipeline).
