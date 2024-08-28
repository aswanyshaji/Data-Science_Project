import sqlite3
import pandas as pd

# Define the database and table names
database_name = 'qa_database.db'  # Ensure your database file has the correct extension (.db)
table_name = 'qa_data'
output_csv_file = 'qa_data.csv'   # Name of the output CSV file

# Connect to the SQLite database
conn = sqlite3.connect(database_name)

# Read the table into a pandas DataFrame
df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

# Close the database connection
conn.close()

# Write the DataFrame to a CSV file
df.to_csv(output_csv_file, index=False)

print(f"Data from {table_name} table has been written to {output_csv_file} successfully.")