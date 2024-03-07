#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
 


# In[ ]:





# In[3]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd

class TestCleaningPipeline(BaseEstimator, TransformerMixin):
    
    def __init__(self, day_columns, large_continuous_variables, non_binary_vars, normalization_stats, num_std=3, small_value=1e-6, stats_train=None, train_numeric_columns=None, train_filtered_columns=None):
        self.day_columns = day_columns
        self.num_std = num_std
        self.small_value = small_value
        self.scaler = StandardScaler()
        self.non_binary_vars = non_binary_vars
        self.normalization_stats = normalization_stats
        self.large_continuous_variables = large_continuous_variables
        self.stats_train = stats_train
        self.train_numeric_columns = train_numeric_columns
        self.train_filtered_columns = train_filtered_columns

    def fit(self, X, y=None):
        # La méthode fit ne fait rien car le scaler est ajusté dans transform
        return self

    def convert_days_to_years_and_rename(self, df, columns):
        df_copy = df.copy()
        converted_columns = {}
        
        for col in columns:
            # Vérifiez si la colonne existe dans df_copy avant de continuer
            if col in df_copy.columns:
                new_col_name = col.replace('DAYS_', 'YEARS_')
                # Convertir les jours en années, en évitant les valeurs aberrantes pour DAYS_EMPLOYED
                if col == 'DAYS_EMPLOYED':
                    converted_columns[new_col_name] = df_copy[col].apply(lambda x: x / -365 if x < 0 else 0)
                else:
                    converted_columns[new_col_name] = df_copy[col].apply(lambda x: x / -365)
                # Supprimez la colonne d'origine après la conversion
                df_copy.drop(columns=[col], inplace=True)
        
        # Ajouter les nouvelles colonnes converties
        df_converted = pd.concat([df_copy, pd.DataFrame(converted_columns)], axis=1)
        
        return df_converted


    def encode_dataframe(self, data):
        # Sélectionnez les colonnes catégorielles pour l'encodage
        categorical_columns = data.select_dtypes(exclude='number').columns
        # Créez un encodeur OneHotEncoder
        cat_transformer = OneHotEncoder(handle_unknown='ignore')
    
        # Adapter l'encodeur aux données et transformer
        encoded_categorical = cat_transformer.fit_transform(data[categorical_columns]).toarray()
    
        # Créez un DataFrame à partir des colonnes encodées
        encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=cat_transformer.get_feature_names_out(categorical_columns))
        encoded_categorical_df.index = data.index
    
        # Sélectionnez les colonnes numériques
        numeric_columns = [col for col in data.columns if col not in categorical_columns]
    
        # Concaténez les colonnes encodées catégorielles avec les colonnes numériques
        encoded_data = pd.concat([encoded_categorical_df, data[numeric_columns]], axis=1)
    
        return encoded_data

    def align_test_with_train(self, test_encoded, columns, colsnum_train, target):
    
        # Supprimer les variables dont toutes les valeurs sont manquantes
        cat_columns_with_nan = test_encoded.columns[test_encoded.isna().all()]
        test_encoded = test_encoded.drop(columns=cat_columns_with_nan)
    
        # Remplacement des valeurs manquantes dans les colonnes numériques de test
        for colnum in colsnum_train:
            if colnum not in test_encoded.columns and colnum != target:
                # Remplacez les valeurs manquantes par NaN
                test_encoded[colnum] = np.nan
    
        # Traitement des colonnes catégorielles
        for col in columns:
            if col not in test_encoded.columns and col != target:
                test_encoded[col] = 0
    
        # Retirez les colonnes de test_encoded qui ne sont pas dans columns (sauf 'TARGET')
        columns_to_remove = [col for col in test_encoded.columns if col not in columns and col != target]
        test_encoded = test_encoded.drop(columns=columns_to_remove, axis=1)
    
        return test_encoded


    def impute_encoded_test(self, encoded_test, stats_df):
        """
        Impute missing values in the one-hot encoded test DataFrame. Uses default values from stats_df for numeric columns
        and fills missing one-hot encoded columns with zeros.
    
        Parameters:
        - encoded_test: DataFrame containing the test data after one-hot encoding.
        - stats_df: DataFrame containing default values for numeric columns.
    
        Returns:
        - DataFrame with imputed values.
        """
        # Iterating through each column in the encoded test DataFrame
        for col in encoded_test.columns:
            if col in stats_df.columns:
                # If the column is numeric and has a default value in stats_df, use that value to fill missing values
                encoded_test[col].fillna(stats_df[col].iloc[0], inplace=True)
            else:
                # If the column is a result of one-hot encoding and does not exist in stats_df, fill missing values with 0
                if encoded_test[col].isnull().any():
                    encoded_test[col].fillna(0, inplace=True)
                    
        return encoded_test  

    def symmetrical_log_transform(self, df, list_vars, small_value):
        """
        Effectue une transformation logarithmique symétrique sur les variables spécifiées dans un DataFrame.
    
        :param df: DataFrame contenant les données.
        :param list_vars: Liste des noms des variables à transformer.
        :param small_value: Petite valeur ajoutée aux données avant la transformation logarithmique.
        :return: Nouveau DataFrame avec uniquement les variables transformées.
        """
        # Créez une copie du DataFrame pour conserver uniquement les variables transformées
        df_transformed = df.copy()
        
        for var in list_vars:
            if var in df_transformed.columns:
                # Appliquer la transformation logarithmique symétrique
                df_transformed[var + '_log'] = np.sign(df_transformed[var]) * np.log(np.abs(df_transformed[var]) + small_value)
        
        # Supprimer les variables d'origine du DataFrame transformé
        df_transformed.drop(columns=list_vars, inplace=True)
        
        return df_transformed
        

    def remove_outliers_using_stats(self, data, columns, normalization_stats, num_std=3):
        """
        Supprime les observations avec des outliers dans les variables spécifiées d'un DataFrame pandas,
        en utilisant les statistiques fournies.
        
        :param data: DataFrame pandas à nettoyer.
        :param columns: Liste des colonnes à vérifier pour les outliers.
        :param normalization_stats: Dictionnaire contenant les moyennes et écarts-types pour chaque variable.
        :param num_std: Nombre d'écarts-types utilisé pour définir un outlier. Par défaut à 3.
        :return: DataFrame pandas nettoyé.
        """
        for column in columns:
            if column in normalization_stats:
                # Récupérer la moyenne et l'écart-type pour la variable courante
                mean = normalization_stats[column]['mean']
                std = normalization_stats[column]['std']
                
                # Définir les limites pour identifier les outliers
                lower_limit = mean - (num_std * std)
                upper_limit = mean + (num_std * std)
                
                # Filtrer les données pour supprimer les outliers
                data = data[(data[column] >= lower_limit) & (data[column] <= upper_limit)]
            
        return data

    
    def normalize_test_variables(self, test_df, non_binary_vars, normalization_stats):
        """
        Normalise les variables non binaires du DataFrame de test en utilisant les statistiques fournies.
    
        Parameters:
        - test_df : DataFrame contenant les données de test.
        - non_binary_vars : Liste des variables non binaires à normaliser.
        - normalization_stats : Dictionnaire contenant les moyennes et écarts-types pour chaque variable non binaire.
    
        Returns:
        - DataFrame de test avec les variables non binaires normalisées.
        """
        # Copie du DataFrame pour éviter de modifier l'original
        test_df_normalized = test_df.copy()
    
        for var in non_binary_vars:
            if var in test_df_normalized.columns:
                # Récupérer la moyenne et l'écart-type pour la variable courante
                mean = normalization_stats[var]['mean']
                std = normalization_stats[var]['std']
                
                # Appliquer la normalisation
                test_df_normalized[var] = (test_df_normalized[var] - mean) / std
                
        return test_df_normalized


    def transform(self, X, y=None):

        # Convertir les variables days en years et renommer
        X= self.convert_days_to_years_and_rename(X, self.day_columns)
        
        X = self.encode_dataframe(X)
     
        X = self.align_test_with_train( X, self.train_filtered_columns, self.train_numeric_columns, 'TARGET')
        
        X = self.impute_encoded_test(X, self.stats_train)
        
        sk_id_curr=X['SK_ID_CURR']
        
        # S'ssurer de ne pas inclure SK_ID_CURR dans les transformations
        X.drop(columns=['SK_ID_CURR'], inplace=True)
    
        # Convertir les variables days en years et renommer
        X= self.convert_days_to_years_and_rename(X, self.day_columns)
    
        # Appliquer la transformation logarithmique symétrique
        X = self.symmetrical_log_transform(X, self.large_continuous_variables, self.small_value)
       
        X = self.remove_outliers_using_stats(X, self.large_continuous_variables, self.normalization_stats, self.num_std)
       
        # Normaliser les variables non binaires
        X= self.normalize_test_variables(X, self.non_binary_vars, self.normalization_stats)
        
        # Réintégrer SK_ID_CURR au DataFrame transformé
        X_transformed = pd.concat([sk_id_curr.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
        
        return X_transformed

# In[ ]:




