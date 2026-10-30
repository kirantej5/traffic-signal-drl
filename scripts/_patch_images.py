import re, sys

path = r"e:\Github projects\traffic-signal-drl\scripts\generate_images.py"
with open(path, encoding='ascii') as f:
    c = f.read()

# Fix get_cmap('name', N) -> matplotlib.colormaps['name']  (N arg dropped, handled via calling cmap(i))
c = re.sub(r"matplotlib\.colormaps\.get_cmap\('(\w+)',\s*\d+\)", r"matplotlib.colormaps['\1']", c)
# Fix get_cmap('name') -> matplotlib.colormaps['name']
c = re.sub(r"matplotlib\.colormaps\.get_cmap\('(\w+)'\)", r"matplotlib.colormaps['\1']", c)
# Fix plt.cm.ScalarMappable -> matplotlib.cm.ScalarMappable (already done but ensure)
c = c.replace("plt.cm.ScalarMappable", "matplotlib.cm.ScalarMappable")
# Fix plt.Normalize -> matplotlib.colors.Normalize
c = c.replace("plt.Normalize", "matplotlib.colors.Normalize")
# Ensure matplotlib.colors imported
if "import matplotlib.colors" not in c:
    c = c.replace("import matplotlib\n", "import matplotlib\nimport matplotlib.colors\n")

with open(path, 'w', encoding='ascii') as f:
    f.write(c)
print("Patch applied.")
