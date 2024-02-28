#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Après le reset de l'état d'exécution, les imports doivent être refaits
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample
import xgboost as xgb
import pandas as pd
import numpy as np
import time
import warnings
import ipytest
import pytest
import os

warnings.filterwarnings("ignore")


# In[2]:


def balance_data(df, target_column='TARGET', id_column='SK_ID_CURR', test_size=0.2, taille=None, random_state=42):
    """
    Prépare les données en appliquant le suréchantillonnage SMOTE sur l'ensemble d'entraînement et ajuste la taille finale
    de l'ensemble d'entraînement ainsi que de l'ensemble de test. Permet également de réduire la taille de l'ensemble d'entraînement
    équilibré à une taille spécifique.

    Paramètres:
    - df : DataFrame contenant les données.
    - target_column : nom de la colonne cible.
    - test_size : proportion de l'ensemble de test après réduction.
    - train_size : taille souhaitée de l'ensemble d'entraînement après équilibrage (avant réduction).
    - taille : taille finale souhaitée de l'ensemble d'entraînement équilibré après réduction.
    - random_state : graine pour la reproductibilité.

    Retourne:
    - X_train_balanced, y_train_balanced : données d'entraînement équilibrées et ajustées à la taille spécifiée.
    - X_test_reduced, y_test_reduced : données de test réduites selon test_size.
    """
    
    # Exclure la colonne cible et éventuellement la colonne d'identifiant
    if id_column:
        X = df.drop([target_column, id_column], axis=1)
    else:
        X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Fractionnement initial pour créer un ensemble de test non touché
    X_train_initial, X_test_initial, y_train_initial, y_test_initial = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Application de SMOTE sur l'ensemble d'entraînement
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_initial, y_train_initial)
    
    # Réduction de l'ensemble d'entraînement équilibré à la taille spécifiée
    if taille and len(X_train_balanced) > taille:
        X_train_balanced, y_train_balanced = resample(X_train_balanced, y_train_balanced,
                                                      replace=False, n_samples=taille,
                                                      random_state=random_state)

    # Réduction de l'ensemble de test si nécessaire
    if test_size < 1.0:
        X_test_reduced, _, y_test_reduced, _ = train_test_split(X_test_initial, y_test_initial, test_size=test_size, random_state=random_state)
    else:
        X_test_reduced, y_test_reduced = X_test_initial, y_test_initial

    return X_train_balanced, y_train_balanced, X_test_reduced, y_test_reduced



# In[3]:


def business_cost(y_true, y_pred, cost_fn, cost_fp):
    """
    Calcule le coût métier basé sur les faux négatifs et les faux positifs.
    """
    fn = sum((y_pred == 0) & (y_true == 1))
    fp = sum((y_pred == 1) & (y_true == 0))
    return fn * cost_fn + fp * cost_fp

def business_score_metric(y_true, y_pred, cost_fn, cost_fp):
    """
    Métrique personnalisée qui calcule le business score pour la validation croisée.
    """
    fn = sum((y_pred == 0) & (y_true == 1))
    fp = sum((y_pred == 1) & (y_true == 0))
    return - (fn * cost_fn + fp * cost_fp)  # Négatif car GridSearchCV cherche à minimiser la métrique

def find_optimal_threshold(y_test, y_scores, cost_fn, cost_fp):
    """
    Trouve le seuil optimal pour la classification.

    Paramètres :
    - y_test : valeurs réelles
    - y_scores : scores de probabilité prédits par le modèle
    - cost_fn : coût d'un faux négatif
    - cost_fp : coût d'un faux positif

    Retourne :
    - seuil optimal pour la classification
    """
    thresholds = np.linspace(0, 1, 100)
    costs = []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        cost = business_cost(y_test, y_pred, cost_fn, cost_fp)
        costs.append(cost)
    
    # Trouver le seuil avec le coût le plus bas
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold



# In[4]:


class ModelPipeline:
    def __init__(self, models, param_grids, scoring='roc_auc', cost_fn=10, cost_fp=1, test_size=0.2, taille=None, random_state=42):
        """
        Initialisation du pipeline.

        :param models: Liste des modèles de machine learning à évaluer.
        :param param_grids: Liste des dictionnaires contenant les grilles d'hyperparamètres correspondant à chaque modèle.
        :param cost_fn: Coût associé à un faux négatif.
        :param cost_fp: Coût associé à un faux positif.
        """
        self.models = models
        self.param_grids = param_grids
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp
        self.test_size=test_size
        self.taille=taille
        self.random_state=random_state
        self.results = []  # Pour stocker les résultats de chaque modèle
        self.scoring = scoring
        
    

    def optimize_hyperparameters(self, model, param_grid, scoring, X_train, y_train):
        """
        Fonction pour optimiser les hyperparamètres d'un modèle donné.

        :param model: Modèle à optimiser.
        :param param_grid: Grille des hyperparamètres à tester.
        :param X_train: Données d'entraînement.
        :param y_train: Étiquettes d'entraînement.
        :return: Meilleur modèle après optimisation.
        """
        # Si une grille d'hyperparamètres est fournie, utilisez GridSearchCV pour trouver le meilleur ensemble d'hyperparamètres
        if param_grid:
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, n_jobs=-1, cv=5, verbose=1)
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_, grid_search.best_params_
        else:
            # Sinon, entraînez simplement le modèle avec les hyperparamètres par défaut
            model.fit(X_train, y_train)
            return model, model.get_params()

    
    def train_and_evaluate(self, model, X_train, y_train, X_test, y_test, param_grid):
        """
        Fonction pour entraîner et évaluer un modèle.

        :param model: Modèle à évaluer.
        :param X_train: Données d'en traînement.
        :param y_train: Étiquettes d'entraînement.
        :param X_test: Données de test.
        :param y_test: Étiquettes de test.
        :param param_grid: Grille d'hyperparamètres pour l'optimisation.
        """
        start_time = time.time()  # Début du chronométrage de l'entraînement et de l'évaluation
        
            
        best_model, best_params = self.optimize_hyperparameters(model, param_grid, self.scoring, X_train, y_train)
        
        # Évaluation du modèle et calcul des métriques
        start_time_prediction = time.time()  # Début du chronométrage de prédiction
        y_scores = best_model.predict_proba(X_test)[:, 1]
        prediction_time = time.time() - start_time_prediction  # Calcul du temps de prédiction
        optimal_threshold = find_optimal_threshold(y_test, y_scores, self.cost_fn, self.cost_fp)
        y_pred_optimal = (y_scores >= optimal_threshold).astype(int)
        business_score = business_cost(y_test, y_pred_optimal, self.cost_fn, self.cost_fp)
        accuracy = accuracy_score(y_test, y_pred_optimal)
        auc_score = roc_auc_score(y_test, y_scores)
        
        execution_time = time.time() - start_time  # Calcul du temps d'exécution total
        
        # Ajout des résultats dans un DataFrame
        results = {
            'Model': type(model).__name__,
            'Accuracy': accuracy,
            'AUC': auc_score,
            'Business Score': business_score,
            'Optimal Threshold': optimal_threshold,
            'Execution Time': execution_time,
            'Prediction Time': prediction_time,
            'Best Model': best_model  # Stockage du modèle pour une utilisation ultérieure
        }
        self.results.append(results)

    def run_pipeline(self, X, target_column, id_column):
        """
        Exécute le pipeline pour tous les modèles fournis.

        :param X_train: Données d'entraînement.
        :param y_train: Étiquettes d'entraînement.
            :param X_test: Données de test.
        :param y_test: Étiquettes de test.
        """
        # Séparation, équilibrage et réduction
        X_train, y_train, X_test, y_test = balance_data(X, target_column, id_column, test_size=self.test_size, taille=self.taille, random_state=self.random_state)
        
        for model, param_grid in zip(self.models, self.param_grids):
            print(f"Entraînement et évaluation du modèle : {type(model).__name__}")
            self.train_and_evaluate(model, X_train, y_train, X_test, y_test, param_grid)
        
        # Après l'exécution de tous les modèles, convertir les résultats en DataFrame pour une analyse facile
        results_df = pd.DataFrame(self.results)
        return results_df
    
def get_best_model_with_threshold(results_df, metric):
    """
    Sélectionne le meilleur modèle basé sur une métrique spécifique et retourne le seuil optimal.

    :param results_df: DataFrame contenant les résultats de l'évaluation de chaque modèle.
    :param metric: La métrique sur laquelle baser la sélection du meilleur modèle.
    :return: Tuple contenant le meilleur modèle et son seuil optimal.
    """
    best_row = results_df.loc[results_df[metric].idxmin()]
    best_model = best_row['Best Model']
    optimal_threshold = best_row['Optimal Threshold']
    return best_model, optimal_threshold


random_state=42

models = [
    xgb.XGBClassifier(random_state=random_state, n_jobs=-1),
    LogisticRegression(random_state=random_state, solver='saga', max_iter=5000, n_jobs=-1),
]

# Grille d'hyperparamètres simple pour XGBClassifier
xgb_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.2]
}

# Grille d'hyperparamètres simple pour LogisticRegression
lr_param_grid = {
    'C': [0.1, 1.0],
    'penalty': ['l1', 'l2']
}

param_grids = [
  xgb_param_grid, lr_param_grid  
]


# In[5]:



# Obtenez le chemin absolu du répertoire actuel du fichier de test
current_directory = os.getcwd()

@pytest.fixture(scope="function")
def test_pipeline_execution():
    # Construire le chemin absolu vers train_sample.csv en utilisant un chemin relatif
    csv_file_path = "train_sample.csv"
    
    pipeline = ModelPipeline(models, param_grids, scoring='roc_auc', cost_fn=10, cost_fp=1, test_size=0.2, taille=5000, random_state=42)
    results_df = pipeline.run_pipeline(pd.read_csv(csv_file_path), 'TARGET', 'SK_ID_CURR')
    return results_df
    

def test_pipeline_results(test_pipeline_execution):
    results_df = test_pipeline_execution
    assert not results_df.empty, "Le DataFrame des résultats est vide"

    # Vérifier les plages pour Accuracy, AUC et Optimal Threshold
    assert all((0 <= results_df['Accuracy']) & (results_df['Accuracy'] <= 1)), "La métrique Accuracy est en dehors de la plage [0, 1]"
    assert all((0 <= results_df['AUC']) & (results_df['AUC'] <= 1)), "La métrique AUC est en dehors de la plage [0, 1]"
    assert all((0 <= results_df['Optimal Threshold']) & (results_df['Optimal Threshold'] <= 1)), "La métrique Optimal Threshold est en dehors de la plage [0, 1]"

    # Vérifier si le nombre d'entrées dans results_df correspond au nombre d'observations dans X_train_test
    assert len(results_df) == len(models), "Incohérence dans le nombre d'entrées entre results_df et X_train_test"










