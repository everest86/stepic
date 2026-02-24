import pandas as pd

# rows = []
#
# with open("pmap.txt") as f:
#     for line in f:
#         line = line.rstrip()
#         if not line or line.startswith("Address"):
#             continue
#
#         parts = line.split(None, 5)
#         if len(parts) < 5:
#             continue
#
#         address = parts[0]
#         kbytes = int(parts[1])
#         rss = int(parts[2])
#         dirty = int(parts[3])
#         mode = parts[4]
#         mapping = parts[5] if len(parts) == 6 else ""
#
#         rows.append([address, kbytes, rss, dirty, mode, mapping])
#
# df = pd.DataFrame(
#     rows,
#     columns=["address", "kbytes", "rss", "dirty", "mode", "mapping"]
# )

process=pd.read_csv('pmap.csv')
process.drop('address', axis=1, inplace=True)
# process.drop('Unnamed', axis=1, inplace=True)
rss_mem=process.groupby('mapping')['rss'].sum().sort_values(ascending=False)
print(rss_mem)