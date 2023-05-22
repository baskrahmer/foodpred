wandb sweep --project harrygobert sweep.yaml
read -p "Enter Sweep ID: " SWEEP_ID
rm nohup.out
nohup wandb agent baskra/harrygobert/$SWEEP_ID --count 100 &
tail -f nohup.out