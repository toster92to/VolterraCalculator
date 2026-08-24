import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor

file_paths = [
    r"/home/mrani/Volterra/Final/Volterra_Input_Data/bat15_2h.csv",
    r"/home/mrani/Volterra/Final/Volterra_Input_Data/bat15_2m.csv",
    r"/home/mrani/Volterra/Final/Volterra_Input_Data/bat15_2l.csv"
]

def process_file(file_path, chunk_size=100000):
    start_time = time.time()

    issues_summary = {
        'total_rows': 0,
        'corrupt_rows': 0,
        'nan_count': 0,
        'missing_values_count': 0,
        'fixed_values_count': 0
    }

    # Create a list to store cleaned chunks
    cleaned_chunks = []

    # Read the file in chunks
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, header=None):
        issues_summary['total_rows'] += len(chunk)

        # Check for rows with incorrect number of columns and remove them
        corrupt_rows = chunk[chunk.apply(lambda row: len(row.dropna()) != 3, axis=1)]
        issues_summary['corrupt_rows'] += len(corrupt_rows)
        chunk = chunk[chunk.apply(lambda row: len(row.dropna()) == 3, axis=1)]

        # Fix NaN values and count them
        nan_count = chunk.isna().sum().sum()
        issues_summary['nan_count'] += nan_count
        issues_summary['fixed_values_count'] += nan_count
        chunk = chunk.fillna(0)

        # Fix missing values
        missing_values_count = chunk.isnull().sum().sum()
        issues_summary['missing_values_count'] += missing_values_count
        issues_summary['fixed_values_count'] += missing_values_count
        chunk = chunk.fillna(0)

        # Store cleaned chunk
        cleaned_chunks.append(chunk)

    # Combine all cleaned chunks into a single DataFrame
    cleaned_data = pd.concat(cleaned_chunks, ignore_index=True)

    # Save cleaned data to a new CSV file
    output_file_path = file_path.replace(".csv", "_fixed.csv")
    cleaned_data.to_csv(output_file_path, index=False, header=False)

    end_time = time.time()

    # Print the summary of issues and processing time
    print(f"File path: {file_path}")
    print(f"Processing time: {end_time - start_time} seconds")
    print("Summary of issues:")
    print(f"Total rows processed: {issues_summary['total_rows']}")
    print(f"Total corrupt rows (removed): {issues_summary['corrupt_rows']}")
    print(f"Total NaN values fixed: {issues_summary['nan_count']}")
    print(f"Total missing values fixed: {issues_summary['missing_values_count']}")
    print(f"Total values fixed: {issues_summary['fixed_values_count']}")
    print("\n")

# Process each file in parallel
if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=24) as executor:
        executor.map(process_file, file_paths)
