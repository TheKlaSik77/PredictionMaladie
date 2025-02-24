import pandas as pd
from sklearn.preprocessing import LabelEncoder

maladies_training = pd.read_csv("./data/Training.csv")
maladies_test = pd.read_csv("./data/Testing.csv")

pd.set_option('display.max_columns', None)  # Afficher toutes les colonnes
pd.set_option('display.expand_frame_repr', False)  # Empêcher de couper les colonnes sur plusieurs lignes
pd.set_option('display.max_colwidth', None)  # Ne pas tronquer les contenus trop longs

liste_nom_fichier = ["maladies_training", "maladies_test"]
index_nom_fichier = 0
for maladies in [maladies_training,maladies_test]:
    # ---------------------- On crée un index -------------------------------
    maladies['maladie_id'] = maladies.index
    maladies.set_index('maladie_id', inplace=True)
    # ------------------------------------------------------------------------
    # On liste toutes les maladies possibles
    liste_maladies = maladies['prognosis'].unique()

    # ------------- Supression de la colonne 'Unnamed: 133 --------------------------

    if 'Unnamed: 133' in maladies.columns :
        maladies = maladies.drop(columns=['Unnamed: 133'])

    # On vérifie que plus aucune colonne ne contienne de valeur vide
    if maladies.isnull().sum().sum() == 0 :
        print("Aucune colonne contient de null\n")
    else :
        print("Une ou plusieurs colonnes contiennent de null\n")

    print(f"Nombre de valeurs manquantes pour chaque colonne : \n{maladies.isnull().sum()}")

    # ------------------------ Encodage prognosis -------------------------------------------

    label_encoders = {}

    label_encoder = LabelEncoder()
    maladies['prognosis'] = label_encoder.fit_transform(maladies['prognosis'].astype(str))
    label_encoders['prognosis'] = label_encoder

    print(maladies['prognosis'].unique())

    # ------------------------ Export avec nom fichier correspondant ----------------------------
    maladies.to_csv(f'./data/{liste_nom_fichier[index_nom_fichier]}_processed.csv')
    index_nom_fichier += 1