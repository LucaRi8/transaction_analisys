import numpy as np 
# associations between catecories and subcategories
categories_associations = {
    "Casa": ["Affitto", "Bollette", "Pulizia", np.nan],
    "Spesa": [np.nan],
    "Viaggi/Esperienze": [np.nan],
    "Trasporti": ["Benzina", "Mezzi pubblici", "Mezzi a noleggio", "Autostrada", np.nan],
    "Serate": ["Alcool", "Entrate/Biglietti", np.nan],
    "Pasti fuori": ["Cene", "Pranzi", "Pranzi lavoro", np.nan],
    "Sport": ["Pass", "Attrezzatura", np.nan],
    "Abbonamenti": [np.nan],
    "Shopping": [np.nan],
    "Stipendio": [np.nan],
    "Altre entrate": [np.nan],
    "Giroconto entrata": [np.nan],
    "Giroconto uscita": [np.nan],
    "Altro": [np.nan]
}

cols_to_keep = [
    "DATA",
    "GIORNO",
    "MESE",
    "ANNO",
    "TIPO TRANSAZIONE",
    "CONTO",
    "CATEGORIA",
    "SOTTOCATEGORIA",
    "IMPORTO"
]

tipo_conto_ass = {
    'Uscita' : [
        "Casa",
        "Spesa",
        "Viaggi/Esperienze",
        "Trasporti",
        "Serate",
        "Pasti fuori",
        "Sport",
        "Abbonamenti",
        "Shopping",
        "Altro"
    ],
    'Giroconto' : [
        'Giroconto uscita',
        'Giroconto entrata'
    ],
    'Entrata' : ['Stipendio',
                 'Altre entrate']
}







