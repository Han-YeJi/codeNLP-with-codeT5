

python main.py --epochs=50 --batch_size=8 --patience=5 --exp_num=11 \
               --pretrained_model=Salesforce/codet5-base --tag=codet5-base

python main.py --epochs=50 --batch_size=8 --patience=5 --exp_num=12 \
              --pretrained_model=Salesforce/codet5-base --tag=codet5-1fold --fold=1

python main.py --epochs=50 --batch_size=8 --patience=5 --exp_num=13 \
              --pretrained_model=Salesforce/codet5-base --tag=codet5-2fold --fold=2

python main.py --epochs=50 --batch_size=8 --patience=5 --exp_num=14 \
              --pretrained_model=Salesforce/codet5-base --tag=codet5-3fold --fold=3

python main.py --epochs=50 --batch_size=8 --patience=5 --exp_num=15 \
              --pretrained_model=Salesforce/codet5-base --tag=codet5-4fold --fold=4

