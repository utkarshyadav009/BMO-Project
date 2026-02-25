import csv

file1 = "metadata.csv"
file2 = "bmo_metadata.csv"

column_name = "filename"

# Read second file into a set
with open(file2, newline='', encoding='utf-8') as f2:
    reader2 = csv.DictReader(f2, delimiter='|')
    file2_values = {row[column_name].strip() for row in reader2}

missing = []

# Compare against first file
with open(file1, newline='', encoding='utf-8') as f1:
    reader1 = csv.DictReader(f1, delimiter='|')
    for row in reader1:
        value = row[column_name].strip()
        if value not in file2_values:
            missing.append(value)

# Print results
if missing:
    print("Files in metadata.csv but NOT in bmo_metadata.csv:")
    for item in missing:
        print(item)
else:
    print("All files from metadata.csv exist in bmo_metadata.csv.")