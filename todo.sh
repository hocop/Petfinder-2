COLOR='\033[1;35m'
NC='\033[0m' # No Color

./setup.sh;

for c in configs/todo/*; do
    name=`echo $c | cut -c 14-`
    command=`echo "python3 scripts/train.py --wandb_project pf2 --config $c --name $name"`
    echo -e "${COLOR}$command${NC}"
    eval $command
done
