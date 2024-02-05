# DDPM without context
python run.py

# sampling without context
# python run.py --mode=sampling --ckpt=./ckpt/DDPM/50_DDPM.pth

# ddim sampling without context
# python run.py --mode=ddim_sampling --ckpt=./ckpt/DDPM/50_DDPM.pth


# train model with context
# python run.py -c --model_name=DDPM_context

# sampling with context
# python run.py -c --model_name=DDPM_context --mode=sampling --ckpt=./ckpt/DDPM_context/50_DDPM_context.pth

# ddim sampling with context
# python run.py -c --model_name=DDPM_context --mode=ddim_sampling --ckpt=./ckpt/DDPM_context/50_DDPM_context.pth