#!/usr/bin/env bash
# Extract all CSVs from raw .ulg logs
echo "Installing pyulog..."
pip install pyulog -q

echo "Extracting .ulg files into .csv formats..."
find "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/raw_logs" -name "*.ulg" | while read ulg_file; do
    out_dir=$(dirname "$ulg_file")
    echo "Extracting: $ulg_file to $out_dir"
    ulog2csv "$ulg_file" -o "$out_dir"
done
echo "Extraction completed."
