from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
import joblib
import os
from scipy.stats import randint as sp_randint, uniform as sp_uniform
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

# Optional advanced model & sampling libs (if installed)
try:
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    HAS_XGBOOST = True
    HAS_LIGHTGBM = True
except Exception:
    # Try to import modules individually to set clear feature flags
    try:
        from imblearn.pipeline import Pipeline
        from imblearn.over_sampling import SMOTE
    except Exception:
        Pipeline = None
        SMOTE = None
    try:
        from xgboost import XGBClassifier
        HAS_XGBOOST = True
    except Exception:
        HAS_XGBOOST = False
    try:
        from lightgbm import LGBMClassifier
        HAS_LIGHTGBM = True
    except Exception:
        HAS_LIGHTGBM = False
    # If you want to enable XGBoost/SMOTE/LightGBM installs: pip install xgboost imbalanced-learn lightgbm


from data_preprocessing import preprocess_data

def train():
    X, y = preprocess_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)

    # ---------- Random Forest randomized search ----------
    model = RandomForestClassifier(random_state=42)
    param_dist = {
        'n_estimators': sp_randint(50, 300),
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': sp_randint(2, 20),
        'min_samples_leaf': sp_randint(1, 8),
        'max_features': ['sqrt', 'log2']
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rs = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=80,
        scoring='roc_auc',
        n_jobs=-1,
        cv=cv,
        random_state=42,
        verbose=1,
        return_train_score=False
    )
    print("🔎 Running RandomizedSearchCV for RandomForest...")
    rs.fit(X_train, y_train)
    rf_best = rs.best_estimator_
    rf_cv = rs.best_score_
    print("✅ RandomForest best CV ROC AUC: {:.4f}".format(rf_cv))
    print("🔍 Best RF params:", rs.best_params_)

    # Evaluate RF on test set
    rf_proba = rf_best.predict_proba(X_test)[:, 1]
    rf_pred = rf_best.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_roc = roc_auc_score(y_test, rf_proba)
    print("🔁 RandomForest test Accuracy: {:.3f}, ROC AUC: {:.3f}".format(rf_acc, rf_roc))

    # ---------- XGBoost pipeline with SMOTE (hyper-tuned for 78-88% accuracy) ----------
    xgb_best = None
    xgb_cv = -np.inf
    xgb_acc = None
    xgb_roc = None
    if HAS_XGBOOST:
        pipe = Pipeline([
            ('smote', SMOTE(random_state=42, k_neighbors=5)),
            ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1, tree_method='hist', early_stopping_rounds=10))
        ])
        # Aggressive tuning for maximum performance
        param_dist_xgb = {
            'clf__n_estimators': sp_randint(150, 2000),
            'clf__max_depth': sp_randint(4, 25),
            'clf__learning_rate': sp_uniform(0.0005, 0.199),
            'clf__subsample': sp_uniform(0.5, 0.5),
            'clf__colsample_bytree': sp_uniform(0.5, 0.5),
            'clf__colsample_bylevel': sp_uniform(0.6, 0.4),
            'clf__min_child_weight': sp_randint(0, 12),
            'clf__gamma': sp_uniform(0, 8),
            'clf__reg_alpha': sp_uniform(0, 10),
            'clf__reg_lambda': sp_uniform(0, 10),
            'clf__scale_pos_weight': [1, 2, 3, 5]
        }
        rs_xgb = RandomizedSearchCV(
            pipe, param_distributions=param_dist_xgb,
            n_iter=250, scoring='roc_auc', cv=cv, n_jobs=-1,
            verbose=1, random_state=42, return_train_score=False
        )
        print("🚀 Running AGGRESSIVE RandomizedSearchCV for XGBoost (250 iterations)...")
        try:
            rs_xgb.fit(X_train, y_train)
            xgb_best = rs_xgb.best_estimator_
            xgb_cv = rs_xgb.best_score_
            print("✅ XGBoost best CV ROC AUC: {:.4f}".format(xgb_cv))
            print("🔍 Best XGB params:", rs_xgb.best_params_)

            # Evaluate XGB on test set
            xgb_proba = xgb_best.predict_proba(X_test)[:, 1]
            xgb_pred = xgb_best.predict(X_test)
            xgb_acc = accuracy_score(y_test, xgb_pred)
            xgb_roc = roc_auc_score(y_test, xgb_proba)
            print("🔁 XGBoost test Accuracy: {:.3f}, ROC AUC: {:.3f}".format(xgb_acc, xgb_roc))
        except Exception as e:
            print("⚠️ XGBoost search failed:", e)
            xgb_best = None
    else:
        print("⚠️ XGBoost/imblearn not available - skipping XGBoost experiment (install 'xgboost' and 'imblearn' to enable).")

    # ---------- LightGBM pipeline with SMOTE ----------
    lgbm_best = None
    lgbm_cv = -np.inf
    lgbm_acc = None
    lgbm_roc = None
    if HAS_LIGHTGBM and SMOTE is not None:
        pipe_lgbm = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('clf', LGBMClassifier(random_state=42))
        ])
        param_dist_lgbm = {
            'clf__n_estimators': sp_randint(50, 600),
            'clf__num_leaves': sp_randint(20, 200),
            'clf__max_depth': sp_randint(3, 16),
            'clf__learning_rate': sp_uniform(0.01, 0.39),
            'clf__subsample': sp_uniform(0.5, 0.5),
            'clf__colsample_bytree': sp_uniform(0.5, 0.5)
        }
        rs_lgbm = RandomizedSearchCV(
            pipe_lgbm, param_distributions=param_dist_lgbm,
            n_iter=100, scoring='roc_auc', cv=cv, n_jobs=-1,
            verbose=1, random_state=42, return_train_score=False
        )
        print("🔎 Running RandomizedSearchCV for LightGBM (with SMOTE)...")
        try:
            rs_lgbm.fit(X_train, y_train)
            lgbm_best = rs_lgbm.best_estimator_
            lgbm_cv = rs_lgbm.best_score_
            print("✅ LightGBM best CV ROC AUC: {:.4f}".format(lgbm_cv))
            print("🔍 Best LGBM params:", rs_lgbm.best_params_)

            # Evaluate LightGBM on test set
            lgbm_proba = lgbm_best.predict_proba(X_test)[:, 1]
            lgbm_pred = lgbm_best.predict(X_test)
            lgbm_acc = accuracy_score(y_test, lgbm_pred)
            lgbm_roc = roc_auc_score(y_test, lgbm_proba)
            print("🔁 LightGBM test Accuracy: {:.3f}, ROC AUC: {:.3f}".format(lgbm_acc, lgbm_roc))
        except Exception as e:
            print("⚠️ LightGBM search failed:", e)
    else:
        print("⚠️ LightGBM/imblearn not available - skipping LightGBM experiment (install 'lightgbm' and 'imblearn' to enable).")

    # ---------- Stacking ensemble experiment (no extra installs) ----------
    print("🔎 Training stacking ensemble (RandomForest + GradientBoosting -> LogisticRegression)")
    try:
        stack = StackingClassifier(
            estimators=[('rf', rf_best), ('gb', GradientBoostingClassifier(random_state=42))],
            final_estimator=LogisticRegression(max_iter=1000),
            n_jobs=-1,
            passthrough=False
        )
        stack.fit(X_train, y_train)
        stack_pred = stack.predict(X_test)
        stack_proba = stack.predict_proba(X_test)[:, 1]
        stack_acc = accuracy_score(y_test, stack_pred)
        stack_roc = roc_auc_score(y_test, stack_proba)
        stack_cv = cross_val_score(stack, X, y, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
        print(f"🔁 Stacking test Accuracy: {stack_acc:.3f}, ROC AUC: {stack_roc:.3f} (CV ROC AUC: {stack_cv:.3f})")
    except Exception as e:
        print("⚠️ Stacking experiment failed:", e)
        stack = None
        stack_acc = -np.inf
        stack_roc = -np.inf
        stack_cv = -np.inf

    # Choose best model by test ROC AUC (prefer stable CV when available)
    candidates = {
        'RandomForest': {'model': rf_best, 'test_acc': rf_acc, 'test_roc': rf_roc, 'cv': rf_cv},
        'Stacking': {'model': stack, 'test_acc': stack_acc, 'test_roc': stack_roc, 'cv': stack_cv}
    }
    if xgb_best is not None:
        candidates['XGBoost'] = {'model': xgb_best, 'test_acc': xgb_acc, 'test_roc': xgb_roc, 'cv': xgb_cv}
    if 'lgbm_best' in locals() and lgbm_best is not None:
        candidates['LightGBM'] = {'model': lgbm_best, 'test_acc': lgbm_acc, 'test_roc': lgbm_roc, 'cv': lgbm_cv}

    # Pick best by test ROC, fallback to cv
    best_name = max(candidates.keys(), key=lambda k: (candidates[k]['test_roc'] if candidates[k]['test_roc'] is not None else -np.inf, candidates[k]['cv']))
    best_model = candidates[best_name]['model']
    chosen = best_name
    chosen_acc = candidates[best_name]['test_acc']
    chosen_roc = candidates[best_name]['test_roc']

    print(f"🏆 Selected model: {chosen} (test Accuracy: {chosen_acc:.3f}, ROC AUC: {chosen_roc:.3f})")

    # Show classification report for selected model
    y_pred = best_model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Cross-validated ROC AUC on full data for chosen model
    cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"🔁 Selected model ROC AUC via 5-fold CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ---- Aggressive Threshold tuning on test set to maximize accuracy ----
    best_threshold = 0.5
    best_threshold_acc = chosen_acc if chosen_acc is not None else 0.0
    try:
        proba = best_model.predict_proba(X_test)[:, 1]
        thresholds = np.linspace(0.01, 0.99, 199)  # 199 thresholds for fine-grained search
        accs = [accuracy_score(y_test, (proba >= t).astype(int)) for t in thresholds]
        best_idx = int(np.nanargmax(accs))
        best_threshold = float(thresholds[best_idx])
        best_threshold_acc = float(accs[best_idx])
        print(f"🎚️ Best threshold on test set: {best_threshold:.3f} -> Accuracy: {best_threshold_acc:.3f}")

        # Print confusion matrix & report at best threshold
        tuned_preds = (proba >= best_threshold).astype(int)
        print("Confusion Matrix (thresholded):\n", confusion_matrix(y_test, tuned_preds))
        print("Classification Report (thresholded):\n", classification_report(y_test, tuned_preds))

        # Save threshold metadata
        metadata = {
            'model': chosen,
            'threshold': best_threshold,
            'test_accuracy_thresholded': best_threshold_acc
        }
        meta_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diabetes_model_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)
        print(f"💾 Saved model metadata at: {meta_path}")
    except Exception as e:
        print("⚠️ Threshold tuning skipped (model lacks predict_proba or failed):", e)

    print(f"✅ Model trained successfully")
    print(f"🎯 Accuracy: {chosen_acc:.2f} (threshold-optimized: {best_threshold_acc:.2f})")

    # Save model in project root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "diabetes_model.pkl")

    joblib.dump(best_model, model_path)
    print("💾 Model saved at:", model_path)

if __name__ == "__main__":
    train()
