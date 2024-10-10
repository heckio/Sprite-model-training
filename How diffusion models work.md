* Diffusion models often rely on a pre-trained model as a starting point for their training process.
* you work through some iteration number using regression adding denoising the image into the final result
* through each iteration noise is added to aid the diffusion process the noise is scaled throughout the iterations util the last one where no noise is added
# Sampling stage

In this stage the function generates samples from noise using a reverse diffusion process with a pre-trained neural network model. It starts with random noise and iteratively refines the noise into a meaningful image. It works by removing predicted noise at each step.
# Here the noise is added and scaled to the current sample 

``` python 
def denoise_add_noise(x, t, pred_noise, z=None):
	""" 
	Denoises the current sample by removing the predicted noise and
	optionally adding random noise back for the reverse diffusion process.
	
	Args: 
		x (torch.Tensor): The current noisy samples (images) to be
		denoised. 
		t (int): The current timestep in the reverse diffusion process.
		pred_noise (torch.Tensor): The noise predicted by the neural
		network at timestep t. 
		z (torch.Tensor): Random noise to add back in for the next step.
		  
	Returns: torch.Tensor: The updated sample after denoising and adding
	noise back. 
	"""
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise 
```

If the is no noise added after training it return the starting noise.

![[Pasted image 20241010025354.png]]

# Results

After 32 epochs it returns images very similar to the starting data set.
![[Pasted image 20241010024930.png]]
 
# Context Sampling

Then the model was sampled using the labels provided on the dataset. 
DDPM was used with a sample size of 25 steps. 


# Acknowledgments

Sprites by [Ebrahim Elgazar](https://www.kaggle.com/datasets/ebrahimelgazar/pixel-art)

This code is modified from, https://github.com/cloneofsimo/minDiffusion and https://learn.deeplearning.ai

Diffusion model is based on [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) and [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)


