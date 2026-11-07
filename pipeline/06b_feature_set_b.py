# Source: S7b.ipynb, code cells [0, 1] (outputs not included)

# %% ---- cell 0 ----
with open(r'<PHD_ROOT>\PHD Research\DATAset\s7\feature_cols.txt') as f:
    cols = [l.strip() for l in f.readlines()]

# Show all unique prefixes to verify group patterns
prefixes = sorted(set(c.split('_')[0] for c in cols))
print("Column prefixes:", prefixes)
print(f"\nTotal: {len(cols)} features")
print("\nAll columns:")
for c in cols:
    print(" ", c)

# %% ---- cell 1 ----

