# MyCode

## DATAFRAME

### Compter le nombre de lignes d'un dataframe

```
rows = len(df)
rows = len(df.axes[0])
rows = df.shape[0]
```

### Compter le nombre d'occurence des valeurs d'une colonne

```
df['col'].value_counts()
```

### Reset index of a dataframe

```
df = df.reset_index()
```

### Rename columns

```
df.columns = ['col1', 'col2']
```
