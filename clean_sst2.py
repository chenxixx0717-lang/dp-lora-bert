import os

def clean_tsv(input_file, output_file):
    """Remove empty lines and ensure consistent format"""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cleaned = []
    for line in lines:
        line = line.strip()
        if line:  # Skip empty lines
            parts = line.split('\t')
            if len(parts) >= 2:  # Ensure at least text and label
                cleaned.append(line)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in cleaned:
            f.write(line + '\n')
    
    print(f"✓ {input_file}: {len(lines)} lines → {len(cleaned)} lines")

# Clean all SST-2 files
clean_tsv('../AdaLoRA-main/NLU/glue_data/SST-2/train.tsv', '../glue_data/SST-2/train_clean.tsv')
clean_tsv('../AdaLoRA-main/NLU/glue_data/SST-2/dev.tsv', '../glue_data/SST-2/dev_clean.tsv')
clean_tsv('../AdaLoRA-main/NLU/glue_data/SST-2/test.tsv', '../glue_data/SST-2/test_clean.tsv')

print("\n✓ All files cleaned")
