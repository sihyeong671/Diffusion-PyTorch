# DDPM PyTorch

**Code Ref**
- [Deeplearning.ai(How Diffusion Model Work)](https://www.deeplearning.ai/short-courses/how-diffusion-models-work/)
- [ML simple works Blog](https://metamath1.github.io/blog/posts/diffusion/ddpm_part2-2.html?utm_source=pytorchkr)
---
## TODO
- [x] sampling animation
- [ ] speed up sampling time
- [x] conditional sampling

---

## How To Use

```sh
# install library
# pip install -r requirementx.txt

# train
python run.py --mode=trian

# sampling
python run.py --mode=sampling
```

---

## Sampling Animation
Jupyter notebook code : [here](./sample/view_sampling.ipynb)

**50 Epoch DDPM**
<p align=center>
    <img width=600px src="./sample/50_epoch_DDPM_animation.gif" loop=infinite/>
</p>

**30 Epoch DDPM**
<p align=center>
    <img width=600px src="./sample/30_epoch_DDPM_animation.gif" loop=infinite/>
</p>

**10 Epoch DDPM**
<p align=center>
    <img width=600px src="./sample/10_epoch_DDPM_animation.gif" loop=infinite/>
</p>

### DDPM with Context
**Hero**
<p align=center>
    <img width=600px src="./sample/50_epoch_DDPM_context_hero_animation.gif" loop=infinite/>
</p>

**Non-Hero**
<p align=center>
    <img width=600px src="./sample/50_epoch_DDPM_context_non-hero_animation.gif" loop=infinite/>
</p>

**Food**
<p align=center>
    <img width=600px src="./sample/50_epoch_DDPM_context_food_animation.gif" loop=infinite/>
</p>

**spell&weapons**
<p align=center>
    <img width=600px src="./sample/50_epoch_DDPM_context_spell-weapon_animation.gif" loop=infinite/>
</p>

**Side Facing**
<p align=center>
    <img width=600px src="./sample/50_epoch_DDPM_context_sideface_animation.gif" loop=infinite/>
</p>

---