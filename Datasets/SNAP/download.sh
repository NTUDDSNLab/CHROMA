#!/bin/bash

# Download the SNAP datasets by the given url and unzip it
# Add dataset url to the following list

# Download dataset name list

dataset_names=(
    "com-livjournal"
    "com-friendster"
    "com-orkut"
    "com-youtube"
    "com-dblp"
    "com-amazon"
    "email-Enron"
    "ca-AstroPh"
    "ca-CondMat"
    "ca-GrQc"
    "ca-HepPh"
    "ca-HepTh"
    "roadNet-CA"
    "roadNet-PA"
    "roadNet-TX"
    "as-skitter"
)


datasets=(
    "https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz"
    "https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz"
    "https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz"
    "https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz"
    "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz"
    "https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz"
    "https://snap.stanford.edu/data/email-Enron.txt.gz"
    "https://snap.stanford.edu/data/ca-AstroPh.txt.gz"
    "https://snap.stanford.edu/data/ca-CondMat.txt.gz"
    "https://snap.stanford.edu/data/ca-GrQc.txt.gz"
    "https://snap.stanford.edu/data/ca-HepPh.txt.gz"
    "https://snap.stanford.edu/data/ca-HepTh.txt.gz"
    "https://snap.stanford.edu/data/roadNet-CA.txt.gz"
    "https://snap.stanford.edu/data/roadNet-PA.txt.gz"
    "https://snap.stanford.edu/data/roadNet-TX.txt.gz"
    "https://snap.stanford.edu/data/as-skitter.txt.gz"

)

echo "Starting download of ${#datasets[@]} datasets..."

# Download and extract each dataset
for i in "${!datasets[@]}"; do
    url="${datasets[$i]}"
    dataset_name="${dataset_names[$i]}"
    
    echo "Downloading dataset $((i+1))/${#datasets[@]}: $dataset_name"
    
    # Extract filename from URL
    filename=$(basename "$url")
    extracted_filename="${filename%.gz}"
    
    # Check if file already exists (either compressed or extracted)
    if [[ -f "$filename" ]] || [[ -f "$extracted_filename" ]]; then
        echo "File already exists, skipping download: $filename"
        continue
    fi
    
    # Download the file
    if wget -O "$filename" "$url"; then
        echo "Successfully downloaded: $filename"
        
        # Extract gz file
        if [[ "$filename" == *.gz ]]; then
            echo "Extracting gz file: $filename"
            gunzip -c "$filename" > "$extracted_filename"
        fi
    else
        echo "Failed to download: $filename"
    fi
    
    echo "---"
done

echo "Download process completed!"
