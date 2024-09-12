#export CUDA_VISIBLE_DEVICES=1
cd ..
#!/bin/bash

# Experiment 2
echo "Starting Experiment 3"

python -m fedavg_mobilnet_poisoned.main poison_fraction=0.5 fraction_mal_clients=0.3
# Check if Experiment 2 succeeded
if [ $? -ne 0 ]; then
  echo "Experiment 3 failed!"
  exit 1
fi

# Experiment 2
echo "Starting Experiment 3"

python -m fedavg_mobilnet_poisoned.main poison_fraction=0.5

# Check if Experiment 2 succeeded
if [ $? -ne 0 ]; then
  echo "Experiment 3 failed!"
  exit 1
fi

# Experiment 2
echo "Starting Experiment 3"

python -m fedavg_mobilnet_poisoned.main poison_fraction=0.3

# Check if Experiment 2 succeeded
if [ $? -ne 0 ]; then
  echo "Experiment 3 failed!"
  exit 1
fi

# Experiment 2
echo "Starting Experiment 3"

python -m fedavg_mobilnet_poisoned.main poison_fraction=0.1

# Check if Experiment 2 succeeded
if [ $? -ne 0 ]; then
  echo "Experiment 3 failed!"
  exit 1
fi

# Experiment 1
echo "Starting Experiment 1 "


python -m fedavg_mobilnet_poisoned.main fraction_mal_clients=0.5

# Check if Experiment 1 succeeded
if [ $? -ne 0 ]; then
  echo "Experiment 1 failed!"
  exit 1
fi

# Experiment 1
echo "Starting Experiment 2 "

python -m fedavg_mobilnet_poisoned.main fraction_mal_clients=0.3

# Check if Experiment 1 succeeded
if [ $? -ne 0 ]; then
  echo "Experiment 1 failed!"
  exit 1
fi

# Experiment 1
echo "Starting Experiment 2 "

python -m fedavg_mobilnet_poisoned.main fraction_mal_clients=0.1

# Check if Experiment 1 succeeded
if [ $? -ne 0 ]; then
  echo "Experiment 1 failed!"
  exit 1
fi


echo "Experiments completed successfully!"
