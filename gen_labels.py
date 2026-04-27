import os
import struct

def text_to_binary_labels(text_file, bin_file):
    """Convert text labels to fairseq binary format"""
    labels = []
    with open(text_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:  # Skip header
                continue
            try:
                label = int(line.strip())
                labels.append(label)
            except ValueError:
                # Skip non-numeric lines (test set has no labels)
                pass
    
    # Write binary format
    with open(bin_file, 'wb') as f:
        f.write(struct.pack('<Q', len(labels)))
        for label in labels:
            f.write(struct.pack('<Q', label))
    
    print(f"✓ {text_file} → {bin_file} ({len(labels)} labels)")

def extract_labels(tsv_file, label_file, skip_header=True):
    """Extract labels from TSV"""
    with open(tsv_file, 'r') as f:
        lines = f.readlines()
    
    with open(label_file, 'w') as f:
        for i, line in enumerate(lines):
            if i == 0 and skip_header:
                f.write('label\n')
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                label = parts[1]
                f.write(label + '\n')
            else:
                # Test set has no labels
                f.write('\n')
    
    print(f"✓ Extracted labels from {tsv_file} → {label_file}")

# Step 1: Extract labels
extract_labels('train_with_canary.tsv', '../glue_data/SST-2-canary-processed/train.label', skip_header=False)
extract_labels('../glue_data/SST-2/dev_clean.tsv', '../glue_data/SST-2-canary-processed/dev.label')
extract_labels('../glue_data/SST-2/test_clean.tsv', '../glue_data/SST-2-canary-processed/test.label')

# Step 2: Convert to binary
os.makedirs('../glue_data/SST-2-canary-bin/label', exist_ok=True)

text_to_binary_labels('../glue_data/SST-2-canary-processed/train.label', 
                      '../glue_data/SST-2-canary-bin/label/train.label')
text_to_binary_labels('../glue_data/SST-2-canary-processed/dev.label', 
                      '../glue_data/SST-2-canary-bin/label/valid.label')
text_to_binary_labels('../glue_data/SST-2-canary-processed/test.label', 
                      '../glue_data/SST-2-canary-bin/label/test.label')

# Step 3: Copy dict
os.system('cp ../glue_data/SST-2-bin/label/dict.txt ../glue_data/SST-2-canary-bin/label/dict.txt')
print("✓ Copied dict.txt")
