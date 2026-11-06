
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter
from pathlib import Path

# Creazione automatica cartella grafici
Path("grafici").mkdir(parents=True, exist_ok=True)

# Impostazioni generali
pd.options.display.float_format = "{:.2f}".format


# ==========================================
# 1. IMPOSTAZIONI GENERALI
# ==========================================

# Visualizzazione valori decimali con due cifre
pd.options.display.float_format = "{:.2f}".format


# ==========================================
# 2. IMPORTAZIONE DATASET
# ==========================================

df = pd.read_csv("data/vendite.csv")


# ==========================================
# 3. ESPLORAZIONE DATASET
# ==========================================

print("\n--- PRIME 5 RIGHE ---")
print(df.head())

print("\n--- DIMENSIONI DATASET ---")
print("Numero righe:", df.shape[0])
print("Numero colonne:", df.shape[1])

print("\n--- INFORMAZIONI GENERALI ---")
df.info()


# ==========================================
# 4. CREAZIONE COLONNA INCASSO
# ==========================================

df["Incasso"] = df["Quantità"] * df["Prezzo_unitario"]

print("\n--- DATASET AGGIORNATO ---")
print(df.head())


# ==========================================
# 5. INCASSO TOTALE DELLA CATENA
# ==========================================

incasso_totale = df["Incasso"].sum()

print("\n--- INCASSO TOTALE ---")
print(f"Incasso totale: € {incasso_totale:.2f}")


# ==========================================
# 6. INCASSO MEDIO PER NEGOZIO
# ==========================================

incasso_negozi = (
    df.groupby("Negozio")["Incasso"]
    .mean()
)

print("\n--- INCASSO MEDIO PER NEGOZIO ---")
print(incasso_negozi)


# ==========================================
# 7. TOP 3 PRODOTTI PIU' VENDUTI
# ==========================================

top_prodotti = (
    df.groupby("Prodotto")["Quantità"]
    .sum()
    .sort_values(ascending=False)
    .head(3)
)

print("\n--- TOP 3 PRODOTTI PIU' VENDUTI ---")
print(top_prodotti)


# ==========================================
# 8. INCASSO MEDIO PER NEGOZIO E PRODOTTO
# ==========================================

analisi_negozio_prodotto = (
    df.groupby(["Negozio", "Prodotto"])["Incasso"]
    .mean()
)

print("\n--- INCASSO MEDIO PER NEGOZIO E PRODOTTO ---")
print(analisi_negozio_prodotto)


# ==========================================
# 9. TABELLA INCASSI
# ==========================================

tabella_incassi = analisi_negozio_prodotto.unstack(fill_value=0)

print("\n--- TABELLA INCASSI ---")
print(tabella_incassi)


# ==========================================
# 10. NUMPY - ESTRAZIONE QUANTITA'
# ==========================================

q = df["Quantità"].to_numpy()

print("\n--- ARRAY QUANTITA' ---")
print(q)


# ==========================================
# 11. NUMPY - STATISTICHE DESCRITTIVE
# ==========================================

media = np.mean(q)
minimo = np.min(q)
massimo = np.max(q)
dev_standard = np.std(q)

print("\n--- STATISTICHE QUANTITA' ---")

print(f"Media: {media:.2f}")
print(f"Minimo: {minimo:.0f}")
print(f"Massimo: {massimo:.0f}")
print(f"Deviazione standard: {dev_standard:.2f}")


# ==========================================
# 12. NUMPY - VENDITE SOPRA LA MEDIA
# ==========================================

sopra_media = q > media

numero_sopra_media = np.sum(sopra_media)

percentuale = (numero_sopra_media / len(q)) * 100

print("\n--- VENDITE SOPRA LA MEDIA ---")

print("Registrazioni sopra la media:", numero_sopra_media)
print(f"Percentuale: {percentuale:.2f}%")


# ==========================================
# 13. NUMPY - ARRAY BIDIMENSIONALE
# ==========================================

dati = np.column_stack((
    df["Quantità"].to_numpy(),
    df["Prezzo_unitario"].to_numpy()
))

print("\n--- MATRICE QUANTITA' E PREZZI ---")

for riga in dati:
    print(f"Quantità: {riga[0]:.0f} | Prezzo: € {riga[1]:.2f}")


# ==========================================
# 14. NUMPY - CALCOLO INCASSI
# ==========================================

incasso_numpy = dati[:, 0] * dati[:, 1]

print("\n--- INCASSI CALCOLATI CON NUMPY ---")
print(incasso_numpy)


# ==========================================
# 15. CONFRONTO PANDAS E NUMPY
# ==========================================

confronto = np.allclose(
    incasso_numpy,
    df["Incasso"].to_numpy()
)

print("\n--- CONTROLLO INCASSI ---")
print("I risultati coincidono:", confronto)


# ==========================================
# 16. MATPLOTLIB - FORMATO MONETARIO
# ==========================================

formato_euro = FuncFormatter(
    lambda x, pos: f"€ {x:,.2f}"
)


# ==========================================
# 17. GRAFICO A BARRE - INCASSI PER NEGOZIO
# ==========================================

incasso_per_negozio = (
    df.groupby("Negozio")["Incasso"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

barre = plt.bar(
    incasso_per_negozio.index,
    incasso_per_negozio.values,
    width=0.45,
    alpha=0.80
)

plt.title("Incasso totale per negozio")
plt.xlabel("Negozio")
plt.ylabel("Incasso (€)")

# Scala verticale più ampia
plt.ylim(0, incasso_per_negozio.max() * 1.25)

# Formattazione monetaria
plt.gca().yaxis.set_major_formatter(formato_euro)

# Griglia orizzontale
plt.grid(axis="y", linestyle="--", alpha=0.35)

# Valori sopra le barre
plt.bar_label(
    barre,
    labels=[
        f"€ {valore:,.2f}"
        for valore in incasso_per_negozio.values
    ],
    padding=5,
    fontsize=9
)

plt.tight_layout()

plt.savefig("grafici/incassi_negozi.png", dpi=300)

plt.show()



# ==========================================
# 18. GRAFICO A TORTA - INCASSI PER PRODOTTO
# ==========================================

incasso_per_prodotto = (
    df.groupby("Prodotto")["Incasso"]
    .sum()
)

plt.figure(figsize=(8, 8))

plt.pie(
    incasso_per_prodotto.values,
    labels=incasso_per_prodotto.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Percentuale incassi per prodotto")

plt.tight_layout()
plt.savefig("grafici/incassi_prodotti.png", dpi=300)
plt.show()


# ==========================================
# 19. GRAFICO A LINEE - ANDAMENTO GIORNALIERO
# ==========================================

incasso_giornaliero = (
    df.groupby("Data")["Incasso"]
    .sum()
)

plt.figure(figsize=(10, 5))

plt.plot(
    incasso_giornaliero.index,
    incasso_giornaliero.values,
    marker="o"
)

plt.title("Andamento giornaliero degli incassi")
plt.xlabel("Data")
plt.ylabel("Incasso (€)")

plt.gca().yaxis.set_major_formatter(formato_euro)

plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.savefig("grafici/andamento_giornaliero.png", dpi=300)
plt.show()


# ==========================================
# 20. CREAZIONE CATEGORIA
# ==========================================

categorie = {
    "Smartphone": "Informatica",
    "Laptop": "Informatica",
    "Tablet": "Informatica",
    "TV": "Elettronica",
    "Smartwatch": "Elettronica"
}

df["Categoria"] = df["Prodotto"].map(categorie)

print("\n--- DATASET CON CATEGORIE ---")
print(df.head())


# ==========================================
# 21. INCASSO TOTALE PER CATEGORIA
# ==========================================

incasso_categoria = (
    df.groupby("Categoria")["Incasso"]
    .sum()
)

print("\n--- INCASSO TOTALE PER CATEGORIA ---")
print(incasso_categoria)


# ==========================================
# 22. QUANTITA' MEDIA PER CATEGORIA
# ==========================================

quantita_media_categoria = (
    df.groupby("Categoria")["Quantità"]
    .mean()
)

print("\n--- QUANTITA' MEDIA PER CATEGORIA ---")
print(quantita_media_categoria)


# ==========================================
# 23. RIEPILOGO PER CATEGORIA
# ==========================================

riepilogo_categoria = (
    df.groupby("Categoria")
    .agg(
        Incasso_totale=("Incasso", "sum"),
        Quantita_media=("Quantità", "mean")
    )
)

print("\n--- RIEPILOGO CATEGORIE ---")
print(riepilogo_categoria)


# ==========================================
# 24. ESPORTAZIONE DATASET AGGIORNATO
# ==========================================

df.to_csv(
    "data/vendite_analizzate.csv",
    index=False,
    float_format="%.2f"
)

print("\nDataset esportato correttamente!")


# ==========================================
# 25. DATI PER GRAFICO COMBINATO
# ==========================================

analisi_grafico = (
    df.groupby("Categoria")
    .agg(
        Incasso_medio=("Incasso", "mean"),
        Quantita_media=("Quantità", "mean")
    )
)

print("\n--- ANALISI MEDIA PER CATEGORIA ---")
print(analisi_grafico)




# ==========================================
# 26. GRAFICO COMBINATO
# ==========================================

fig, ax1 = plt.subplots(figsize=(10, 6))

# Grafico a barre: incasso medio
barre = ax1.bar(
    analisi_grafico.index,
    analisi_grafico["Incasso_medio"],
    width=0.40,
    alpha=0.75,
    label="Incasso medio"
)

ax1.set_xlabel("Categoria")
ax1.set_ylabel("Incasso medio (€)")
ax1.set_title("Incasso medio e quantità media per categoria")

# Scala asse sinistro
ax1.set_ylim(0, 3000)

ax1.yaxis.set_major_formatter(formato_euro)

# Secondo asse verticale
ax2 = ax1.twinx()

# Grafico a linee: quantità media
linea, = ax2.plot(
    analisi_grafico.index,
    analisi_grafico["Quantita_media"],
    color="darkorange",
    linestyle="--",
    linewidth=2.5,
    marker="o",
    markersize=8,
    label="Quantità media",
    zorder=10
)

# Scala asse destro
ax2.set_ylim(0, 10)

ax2.set_ylabel("Quantità media (unità)")

# Manteniamo la linea visibile sopra le barre
ax2.patch.set_visible(False)

# Legenda combinata
ax1.legend(
    [barre, linea],
    ["Incasso medio", "Quantità media"],
    loc="upper left"
)

plt.tight_layout()

plt.savefig("grafici/grafico_combinato.png", dpi=300)

plt.show()

