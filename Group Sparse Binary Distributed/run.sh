#!/bin/bash
#SBATCH --partition=big # Specify the desired partition, e.g. gpu or big
#SBATCH --mem=4G  

echo 'Getting node information'
date;hostname;id;pwd

echo 'Activating virtual environment'
source ~/.bashrc  # Load the .bashrc

echo 'Running script'

for i in {4..8}
do
    echo "Running iteration $i"
    start_time=$(date +%s%N)  # Get start time in nanoseconds
    
    julia --project -p 2 main.jl  # Run the Julia script and save the output

    end_time=$(date +%s%N)  # Get end time in nanoseconds
    elapsed_time=$((end_time - start_time))  # Calculate elapsed time in nanoseconds
    elapsed_time_ms=$((elapsed_time / 1000000))  # Convert to milliseconds
    total_time=$((total_time + elapsed_time_ms))  # Accumulate total time
    
    # Optionally, you can print or save each individual run's time
    echo "Elapsed time for iteration $i: $elapsed_time_ms ms"
done

average_time=$(echo "scale=2; $total_time / 4" | bc)  # Calculate average time
echo "Average time over 4 runs: $average_time ms"
