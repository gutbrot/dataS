# ==========================================
# 0. Alap beállítások, importok
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.2f}".format)  # ne tudományos jelölésben írja

# ==========================================
# 1. Gyors áttekintés
# ==========================================
print("Alak (sor, oszlop):", df.shape)
print("\nTípusok:")
print(df.dtypes)

print("\nElső 5 sor:")
print(df.head())

print("\nInfo:")
print(df.info())

# ==========================================
# 2. Hiányzó értékek, duplikátumok
# ==========================================
print("\nHiányzó értékek oszloponként:")
print(df.isna().sum())

print("\nDuplikált sorok száma:", df.duplicated().sum())

# (Ha akarsz: duplikáltak törlése)
# df = df.drop_duplicates()

# ==========================================
# 3. Típusok finomhangolása (példa a te df-edre)
# ==========================================
# Ha van dátum/év, lehet int -> kategória vagy int marad
df["year_released"] = df["year_released"].astype("int64")

# Ha van kategóriás változó (pl. genre, certificate stb.)
# df["genre"] = df["genre"].astype("category")

# ==========================================
# 4. Leíró statisztika (számszerű oszlopok)
# ==========================================
num_df = df.select_dtypes(include=[np.number])

desc = num_df.describe()

# extra sorok: módusz, ferdeség, csúcsosság
mode_row = num_df.mode().iloc[0]; mode_row.name = "mode"
skew_row = num_df.skew(); skew_row.name = "skew"
kurt_row = num_df.kurt(); kurt_row.name = "kurt"

desc_full = pd.concat(
    [desc, mode_row.to_frame().T, skew_row.to_frame().T, kurt_row.to_frame().T]
)

print("\nSzámszerű leíró statisztika:")
print(desc_full)

# Kategorikus oszlopokra value_counts (példa)
cat_cols = df.select_dtypes(include=["object", "category"]).columns
for col in cat_cols:
    print(f"\nValue counts – {col}:")
    print(df[col].value_counts().head(10))   # top 10 kategória
    print(df[col].value_counts(normalize=True).head(10))  # arányok

# ==========================================
# 5. Univariáns vizsgálat – numerikus eloszlások
# ==========================================
for col in num_df.columns:
    plt.figure(figsize=(8, 4))
    plt.hist(df[col].dropna(), bins=30)
    plt.title(f"{col} – eloszlás")
    plt.xlabel(col)
    plt.ylabel("Gyakoriság")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 3))
    plt.boxplot(df[col].dropna(), vert=False)
    plt.title(f"{col} – boxplot")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# ==========================================
# 6. Univariáns vizsgálat – kategóriák
#    (ha vannak ilyen oszlopok)
# ==========================================
for col in cat_cols:
    plt.figure(figsize=(10, 4))
    df[col].value_counts().head(15).plot(kind="bar")
    plt.title(f"{col} – top 15 kategória")
    plt.xlabel(col)
    plt.ylabel("Darab")
    plt.tight_layout()
    plt.show()

# ==========================================
# 7. Bivariáns vizsgálat – kapcsolat a célváltozóval
#    Tegyük fel, hogy a cél a 'gross' (bevétel)
# ==========================================
target = "gross"

# Numerikus vs target: scatter + korreláció
for col in num_df.columns:
    if col == target:
        continue
    corr = df[[col, target]].corr().iloc[0, 1]
    print(f"Korreláció {col} és {target} között: {corr:.3f}")

    plt.figure(figsize=(8, 4))
    plt.scatter(df[col], df[target], alpha=0.3)
    plt.xlabel(col)
    plt.ylabel(target)
    plt.title(f"{col} vs {target} (corr={corr:.2f})")
    plt.tight_layout()
    plt.show()

# Kategóriás vs target: átlagos target kategóriánként (ha vannak kategóriák)
for col in cat_cols:
    plt.figure(figsize=(10, 4))
    df.groupby(col)[target].mean().sort_values(ascending=False).head(15).plot(kind="bar")
    plt.title(f"Átlagos {target} {col} szerint (top 15)")
    plt.ylabel(f"Átlagos {target}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# ==========================================
# 8. Korrelációs mátrix (számszerű oszlopok)
# ==========================================
corr_mat = num_df.corr()
print("\nKorrelációs mátrix:")
print(corr_mat)

plt.figure(figsize=(8, 6))
plt.imshow(corr_mat, aspect="auto")
plt.colorbar(label="Korreláció")
plt.xticks(range(len(num_df.columns)), num_df.columns, rotation=90)
plt.yticks(range(len(num_df.columns)), num_df.columns)
plt.title("Korrelációs mátrix (numerikus változók)")
plt.tight_layout()
plt.show()

# ==========================================
# 9. Időbeli trendek (specifikus a filmedhez)
#    Bevétel/értékelés az évek szerint
# ==========================================
# Éves átlagos bevétel
yearly_gross = df.groupby("year_released")["gross"].mean()

plt.figure(figsize=(10, 5))
plt.plot(yearly_gross.index, yearly_gross.values)
plt.title("Átlagos bevétel évenként")
plt.xlabel("Megjelenési év")
plt.ylabel("Átlagos bevétel (gross)")
plt.tight_layout()
plt.show()

# Éves átlagos filmrating
yearly_rating = df.groupby("year_released")["movie_rating"].mean()

plt.figure(figsize=(10, 5))
plt.plot(yearly_rating.index, yearly_rating.values)
plt.title("Átlagos filmrating évenként")
plt.xlabel("Megjelenési év")
plt.ylabel("Átlagos rating")
plt.tight_layout()
plt.show()

# ==========================================
# 10. Rövid összefoglaló (ide kézzel írsz majd)
# ==========================================
# - Milyen az eloszlása a bevételnek, ratingnek, szavazatoknak?
# - Vannak-e extrém outlierek (nagyon magas bevétel, nagyon hosszú filmek)?
# - Mely változók korrelálnak legjobban a bevétellel?
# - Mely években a legmagasabb az átlagos bevétel/rating?
