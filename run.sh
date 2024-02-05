# DDPM without context
# python run.py
# sampling without context
# python run.py --mode=sampling --ckpt=./ckpt/DDPM/10_DDPM.pth


# using context
# python run.py -c --model_name=DDPM_context
# sampling using context
python run.py -c --model_name=DDPM_context --mode=sampling --ckpt=./ckpt/DDPM_context/50_DDPM_context.pth
