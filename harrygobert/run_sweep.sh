wandb sweep --project harrygobert sweep.yaml
read -p "Enter Sweep ID: " SWEEP_ID
nohup wandb agent baskra/harrygobert/$SWEEP_ID --count 100 &