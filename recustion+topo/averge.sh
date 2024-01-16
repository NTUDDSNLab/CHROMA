#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 path_to_input_file"
    exit 1
fi

input_file=$1
total_time=0
total_color=0
iteration=100
for (( i=1; i<=${iteration}; i++ )); do
    output=$(./main "$input_file")
    time=$(echo $output | awk '{print $4}')
    color=$(echo $output | awk '{print $7}')
    total_time=$(echo "$total_time + $time" | bc)
    total_color=$(echo "$total_color + $color" | bc)
done

average_time=$(echo "scale=6; $total_time / $iteration" | bc)
average_color=$(echo "scale=6; $total_color / $iteration" | bc)

echo "Average Color Taken: $average_color"
echo "Average GPU Time Taken: $average_time"
