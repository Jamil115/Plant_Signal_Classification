import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


def run_experiment(df, update_status=None):
    result = {
        "models": {},
        "plots": {}
    }

    if update_status:
        update_status("Starting experiment...")

    # Drop Plant_ID if present
    df = df.drop(columns=['Plant_ID'], errors='ignore')
    if update_status:
        update_status("Dropped 'Plant_ID' column if existed.")

    # Save boxplot before outlier removal
    df_melted = df.select_dtypes(include='number').melt(var_name='Feature', value_name='Value')
    fig1, ax1 = plt.subplots(figsize=(14, 10))
    sns.boxplot(x='Feature', y='Value', data=df_melted, color='skyblue', ax=ax1)
    ax1.set_title("Boxplot of All Numeric Features")
    ax1.tick_params(axis='x', rotation=30)
    result['plots']['boxplot_before'] = fig1
    plt.close(fig1)

    # Outlier removal
    for col in df.select_dtypes(include='number').columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    if update_status:
        update_status("Outliers removed.")

    # Save boxplot after outlier removal
    df_melted = df.select_dtypes(include='number').melt(var_name='Feature', value_name='Value')
    fig3, ax3 = plt.subplots(figsize=(14, 10))
    sns.boxplot(x='Feature', y='Value', data=df_melted, color='skyblue', ax=ax3)
    ax3.set_title("After Outlier Removal")
    ax3.tick_params(axis='x', rotation=30)
    result['plots']['boxplot_after'] = fig3
    plt.close(fig3)

    # Split features and target
    x = df.drop(columns=["Plant_Message_Type"])
    y = df["Plant_Message_Type"]

    le = LabelEncoder()
    y = le.fit_transform(y)
    if update_status:
        update_status("Label encoding done.")

    # Heatmap
    df_combined = pd.concat([x, pd.Series(y, name="Plant_Message_Type")], axis=1)
    correlation_matrix = df_combined.corr()
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".4f", cmap='coolwarm', square=True, linewidths=0.5, ax=ax2)
    ax2.set_title("Correlation Heatmap (Features + Target)")
    ax2.tick_params(axis='x', rotation=45)
    result['plots']['correlation_heatmap'] = fig2
    plt.close(fig2)

    # Class Distribution Before SMOTE
    label_map = {0: 'Contentment', 1: 'Distress', 2: 'Invitation', 3: 'Warning'}
    fig_cd_before, ax_cd_before = plt.subplots()
    y_mapped = pd.Series(y).map(label_map)
    sns.countplot(x=y_mapped, ax=ax_cd_before)
    ax_cd_before.set_title("Class Distribution Before SMOTE")
    result['plots']['class_dist_before'] = fig_cd_before
    plt.close(fig_cd_before)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    x_smote, y_smote = smote.fit_resample(x, y)
    if update_status:
        update_status("SMOTE applied.")

    # Class Distribution After SMOTE
    fig_cd_after, ax_cd_after = plt.subplots()
    y_res_mapped = pd.Series(y_smote).map(label_map)
    sns.countplot(x=y_res_mapped, ax=ax_cd_after)
    ax_cd_after.set_title("Class Distribution After SMOTE")
    result['plots']['class_dist_after'] = fig_cd_after
    plt.close(fig_cd_after)

    # Augment with Gaussian noise
    x_smote = pd.DataFrame(x_smote, columns=x.columns)
    float_cols = x_smote.select_dtypes(include=['float32', 'float64']).columns
    noise = np.random.normal(0, 0.001, size=x_smote[float_cols].shape)
    x_noisy = x_smote.copy()
    x_noisy[float_cols] += noise

    x_aug = pd.concat([x_smote, x_noisy], ignore_index=True)
    y_aug = pd.concat([pd.Series(y_smote), pd.Series(y_smote)], ignore_index=True)

    # Train/Test split and scaling
    X_train, X_test, y_train, y_test = train_test_split(x_aug, y_aug, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if update_status:
        update_status("Data split and scaled.")

    def run_grid(model, param_grid, scaled=False):
        if update_status:
            update_status(f"Training {model.__class__.__name__}...")

        grid = GridSearchCV(model, param_grid, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                            scoring='accuracy', n_jobs=-1)
        grid.fit(X_train_scaled if scaled else X_train, y_train)

        if update_status:
            update_status(f"{model.__class__.__name__} training completed.")

        y_pred = grid.best_estimator_.predict(X_test_scaled if scaled else X_test)
        return {
            "best_score": grid.best_score_,
            "best_params": grid.best_params_,
            "test_accuracy": accuracy_score(y_test, y_pred),
            "report": classification_report(y_test, y_pred, target_names=le.classes_, digits=3, output_dict=True),
            "predictions": y_pred
        }

    # Train all models
    result['models']['Random Forest'] = run_grid(RandomForestClassifier(random_state=42), {
        'n_estimators': [100, 200], 'max_depth': [None, 5, 10], 'criterion': ['gini', 'entropy'], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]
    })

    result['models']['XGBoost'] = run_grid(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42), {
        'n_estimators': [100, 200], 'max_depth': [3,6,10], 'learning_rate': [0.001, 0.1, 0.2], 'subsample': [0.8, 1], 'colsample_bytree': [0.8, 1]
    })

    result['models']['SVM'] = run_grid(SVC(probability=True, random_state=42), {
        'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']
    }, scaled=True)

    result['models']['KNN'] = run_grid(KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7, 11], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']
    }, scaled=True)

    if update_status:
        update_status("All models trained successfully!")

    # Confusion matrices 4-in-1
    fig_cm_combined, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    fig_cm_combined.suptitle('Confusion Matrices of Classifiers', fontsize=16, fontweight='bold')

    model_names = ['Random Forest', 'SVM', 'KNN', 'XGBoost']
    cm_colors = ['Blues', 'Oranges', 'Purples', 'Greens']

    for i, model_name in enumerate(model_names):
        y_pred_model = result['models'][model_name]["predictions"]
        cm = confusion_matrix(y_test, y_pred_model)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        disp.plot(ax=axes[i], cmap=cm_colors[i], colorbar=False)
        axes[i].set_title(model_name)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    result['plots']['conf_matrix_all'] = fig_cm_combined
    plt.close(fig_cm_combined)

    return result
