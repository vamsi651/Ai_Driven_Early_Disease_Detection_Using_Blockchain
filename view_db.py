import sqlite3

conn = sqlite3.connect("predictions.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM predictions")
rows = cursor.fetchall()

# Get column names
column_names = [description[0] for description in cursor.description]

# Print header
print(" | ".join(column_names))
print("-" * 120)

# Print rows
for row in rows:
    print(" | ".join(str(value) for value in row))

conn.close()