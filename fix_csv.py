import csv

input_file = 'movie.csv'
output_file = 'movie_fixed.csv'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile, skipinitialspace=True)
    writer = csv.writer(outfile)
    
    # Read header
    try:
        header = next(reader)
        # Ensure header has 5 columns
        header = header[:5]
        writer.writerow(header)
    except StopIteration:
        print("Empty file")
        exit()

    for i, row in enumerate(reader):
        # Skip empty lines or lines with very few columns (likely garbage)
        if not row or len(row) < 2:
            continue
            
        # Clean up row (strip whitespace from fields if any)
        row = [field.strip() for field in row]
        
        # Specific fix for lines that might be indented or weirdly formatted
        # But csv.reader handles split by comma. 
        # The issue with lines 328+ is they have leading empty field if they started with space and comma? 
        # No, they started with spaces then text. "    Inception" -> "    Inception" 
        # csv.reader with default config should handle quoted fields correctly.
        
        # Keep only first 5 columns
        new_row = row[:5]
        
        # If the row has fewer than 5 columns, pad with empty string? 
        # Or better, just filter out bad rows. 
        # Lines 328+ seem to be duplicates of previous lines but formatted poorly.
        # Let's check for duplicates based on Movie Name
        if i >= 327: # Lines 328+ in 1-based index are >= 327 in 0-based index from enumerate (since header consumed)
             # Actually, let's just inspect the content. 
             # The lines 328+ are definitely duplicates and malformed.
             # "    Inception" is not equal to "Inception"
             pass
        
        # Clean the movie name specifically to trim spaces
        if  new_row:
             new_row[0] = new_row[0].strip()
        
        writer.writerow(new_row)

import os
# Replace original file
os.replace(output_file, input_file)
print("Fixed CSV file.")
