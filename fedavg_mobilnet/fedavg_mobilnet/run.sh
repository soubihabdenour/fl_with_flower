
cd ..
#!/bin/bash

# Experiment
echo "Starting Experiment"

python -m fedavg_mobilnet.main dataset.partitioner.name="PathologicalPartitioner"

# Check if Experiment succeeded
if [ $? -ne 0 ]; then
  echo "Experiment 3 failed!"
  exit 1
fi

echo "Experiment completed successfully!"