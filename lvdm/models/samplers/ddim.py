import numpy as np
from tqdm import tqdm
import torch
from lvdm.models.utils_diffusion import make_ddim_sampling_parameters, make_ddim_timesteps
from lvdm.common import noise_like
from diffusers.schedulers import DDIMScheduler
from utils.freeinit_utils import (
    get_freq_filter,
    freq_mix_3d,
)
from einops import rearrange

import scripts.evaluation.ptp_utils as ptp_utils
import abc

class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, hidden_states, video_length, place_in_unet: str):
        video_length = 16
        hidden_states = rearrange(hidden_states, "(b f) d c -> b f d c", f=video_length)
        batch_size = hidden_states.shape[0] // 2

        if batch_size == 2:
            # Do classifier-free guidance
            hidden_states_uncondition, hidden_states_condition = hidden_states.chunk(2)

            if self.cur_step <= self.motion_control_step:
                hidden_states_motion_uncondition = hidden_states_uncondition[
                    1
                ].unsqueeze(0)
            else:
                hidden_states_motion_uncondition = hidden_states_uncondition[
                    0
                ].unsqueeze(0)

            hidden_states_out_uncondition = torch.cat(
                [
                    hidden_states_motion_uncondition,
                    hidden_states_uncondition[1].unsqueeze(0),
                ],
                dim=0,
            )  # Query
            hidden_states_sac_in_uncondition = self.forward(
                hidden_states_uncondition[0].unsqueeze(0), video_length, place_in_unet
            )
            hidden_states_sac_out_uncondition = torch.cat(
                [
                    hidden_states_sac_in_uncondition,
                    hidden_states_uncondition[1].unsqueeze(0),
                ],
                dim=0,
            )  # Key & Value

            if self.cur_step <= self.motion_control_step:
                hidden_states_motion_condition = hidden_states_condition[1].unsqueeze(0)
            else:
                hidden_states_motion_condition = hidden_states_condition[0].unsqueeze(0)

            hidden_states_out_condition = torch.cat(
                [
                    hidden_states_motion_condition,
                    hidden_states_condition[1].unsqueeze(0),
                ],
                dim=0,
            )  # Query
            hidden_states_sac_in_condition = self.forward(
                hidden_states_condition[0].unsqueeze(0), video_length, place_in_unet
            )
            hidden_states_sac_out_condition = torch.cat(
                [
                    hidden_states_sac_in_condition,
                    hidden_states_condition[1].unsqueeze(0),
                ],
                dim=0,
            )  # Key & Value

            hidden_states_out = torch.cat(
                [hidden_states_out_uncondition, hidden_states_out_condition], dim=0
            )
            hidden_states_sac_out = torch.cat(
                [hidden_states_sac_out_uncondition, hidden_states_sac_out_condition],
                dim=0,
            )

        elif batch_size == 1:
            if self.cur_step <= self.motion_control_step:
                hidden_states_motion = hidden_states[1].unsqueeze(0)
            else:
                hidden_states_motion = hidden_states[0].unsqueeze(0)

            hidden_states_out = torch.cat(
                [hidden_states_motion, hidden_states[1].unsqueeze(0)], dim=0
            )  # Query
            hidden_states_sac_in = self.forward(
                hidden_states[0].unsqueeze(0), video_length, place_in_unet
            )
            hidden_states_sac_out = torch.cat(
                [hidden_states_sac_in, hidden_states[1].unsqueeze(0)], dim=0
            )  # Key & Value

        else:
            #  gr.Error(f"Not implemented error")
            raise NotImplementedError
        hidden_states = rearrange(hidden_states, "b f d c -> (b f) d c", f=video_length)
        hidden_states_out = rearrange(
            hidden_states_out, "b f d c -> (b f) d c", f=video_length
        )
        hidden_states_sac_out = rearrange(
            hidden_states_sac_out, "b f d c -> (b f) d c", f=video_length
        )
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
        return hidden_states_out, hidden_states_sac_out, hidden_states_sac_out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0
        self.num_att_layers = -1
        self.motion_control_step = 0
        
    def __init__(self):
        self.cur_step = 0
        self.cur_att_layer = 0
        self.num_att_layers = -1
        self.motion_control_step = 0


class EmptyControl(AttentionControl):
    def forward(self, hidden_states, video_length, place_in_unet):
        return hidden_states


class FreeSAC(AttentionControl):
    def forward(self, hidden_states, video_length, place_in_unet):
        hidden_states_sac = (
            hidden_states[:, 0, :, :].unsqueeze(1).repeat(1, video_length, 1, 1)
        )
        return hidden_states
    
class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.counter = 0
        self.freq_filter = None

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))
        self.use_scale = self.model.use_scale
        print('DDIM scale', self.use_scale)

        if self.use_scale:
            self.register_buffer('scale_arr', to_torch(self.model.scale_arr))
            ddim_scale_arr = self.scale_arr.cpu()[self.ddim_timesteps]
            self.register_buffer('ddim_scale_arr', ddim_scale_arr)
            ddim_scale_arr = np.asarray([self.scale_arr.cpu()[0]] + self.scale_arr.cpu()[self.ddim_timesteps[:-1]].tolist())
            self.register_buffer('ddim_scale_arr_prev', ddim_scale_arr)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               schedule_verbose=False,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        
        # check condition bs
        if conditioning is not None:
            if isinstance(conditioning, dict):
                try:
                    cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                except:
                    cbs = conditioning[list(conditioning.keys())[0]][0].shape[0]

                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=schedule_verbose)
        self.scheduler = DDIMScheduler(num_train_timesteps=S, trained_betas=self.model.betas)

        
        # make shape
        if len(shape) == 3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        elif len(shape) == 4:
            C, T, H, W = shape
            size = (batch_size, C, T, H, W)
        # print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        
        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    verbose=verbose,
                                                    **kwargs)
        return samples, intermediates
    
    @torch.no_grad()
    def init_filter(self, num_channels_latents, video_length, height, width, filter_params = {"method": 'gaussian', "n": 4, "d_s": 1, "d_t": 1}):
        # initialize frequency filter for noise reinitialization
        batch_size = 1
        filter_shape = [
            batch_size,
            num_channels_latents,
            video_length,
            height,
            width,
        ]
        # self.freq_filter = get_freq_filter(filter_shape, device=self._execution_device, params=filter_params)
        self.freq_filter = get_freq_filter(
            filter_shape,
            device=self.model.betas.device,
            filter_type=filter_params["method"],
            n=filter_params["n"] if filter_params["method"] == "butterworth" else None,
            d_s=filter_params["d_s"],
            d_t=filter_params["d_t"],
        )
        
    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, verbose=True,
                      cond_tau=1., target_size=None, start_timesteps=None, num_iters: int = 3, batch_size: int = 1,
                      **kwargs):
        
        # subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
        # timesteps = self.ddim_timesteps[:subset_end]
        # motion_control = 1
        # motion_control_step = motion_control * self.ddim_timesteps.shape[0]
        # attn_controller = FreeSAC()
        # attn_controller.motion_control_step = motion_control_step
        # ptp_utils.register_attention_control(self.model, attn_controller)
        device = self.model.betas.device        
        print('ddim device', device)
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        init_x0 = False
        clean_cond = kwargs.pop("clean_cond", False)
        for iter in range(num_iters):
            if iter == 0:
                initial_noise = img.detach().clone()
            else:
                # 1. DDPM Forward with initial noise, get noisy latents z_T
                # if use_fast_sampling:
                #     current_diffuse_timestep = self.scheduler.config.num_train_timesteps / num_iters * (iter + 1) - 1
                # else:
                #     current_diffuse_timestep = self.scheduler.config.num_train_timesteps - 1
                current_diffuse_timestep = self.scheduler.config.num_train_timesteps - 1 # diffuse to t=999 noise level
                diffuse_timesteps = torch.full((batch_size,),int(current_diffuse_timestep))
                diffuse_timesteps = diffuse_timesteps.long()
                z_T = self.scheduler.add_noise(
                    original_samples=img.to(device), 
                    noise=initial_noise.to(device), 
                    timesteps=diffuse_timesteps.to(device)
                )
                # 2. create random noise z_rand for high-frequency
                z_rand = torch.randn(shape, device=device)
                # 3. Roise Reinitialization
                img = freq_mix_3d(z_T.to(dtype=torch.float32), z_rand, LPF=self.freq_filter)
                
            if timesteps is None:
                timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
            elif timesteps is not None and not ddim_use_original_steps:
                subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
                timesteps = self.ddim_timesteps[:subset_end]
                
            intermediates = {'x_inter': [img], 'pred_x0': [img]}
            time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
            total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
            if verbose:
                iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
            else:
                iterator = time_range

            for i, step in tqdm(enumerate(iterator), total=len(iterator)):
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                if start_timesteps is not None:
                    assert x0 is not None
                    if step > start_timesteps*time_range[0]:
                        continue
                    elif not init_x0:
                        img = self.model.q_sample(x0, ts) 
                        init_x0 = True

                # use mask to blend noised original latent (img_orig) & new sampled latent (img)
                if mask is not None:
                    assert x0 is not None
                    if clean_cond:
                        img_orig = x0
                    else:
                        img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass? <ddim inversion>
                    img = img_orig * mask + (1. - mask) * img # keep original & modify use img
                
                index_clip =  int((1 - cond_tau) * total_steps)
                if index <= index_clip and target_size is not None:
                    target_size_ = [target_size[0], target_size[1]//8, target_size[2]//8]
                    img = torch.nn.functional.interpolate(
                    img,
                    size=target_size_,
                    mode="nearest",
                    )
                outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                        quantize_denoised=quantize_denoised, temperature=temperature,
                                        noise_dropout=noise_dropout, score_corrector=score_corrector,
                                        corrector_kwargs=corrector_kwargs,
                                        unconditional_guidance_scale=unconditional_guidance_scale,
                                        unconditional_conditioning=unconditional_conditioning,
                                        x0=x0,
                                        **kwargs)
                img, pred_x0 = outs
                
                if callback: callback(i)
                if img_callback: img_callback(pred_x0, i)

                if index % log_every_t == 0 or index == total_steps - 1:
                    intermediates['x_inter'].append(img)
                    intermediates['pred_x0'].append(pred_x0)

            timesteps = None
            # attn_controller = EmptyControl()
            # attn_controller.motion_control_step = -1
            # ptp_utils.register_attention_control(self.model, attn_controller)
            # attn_controller.reset()
        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      uc_type=None, conditional_guidance_scale_temporal=None, **kwargs):
        b, *_, device = *x.shape, x.device
        if x.dim() == 5:
            is_video = True
        else:
            is_video = False
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c, **kwargs) # unet denoiser
        else:
            # with unconditional condition
            if isinstance(c, torch.Tensor):
                e_t = self.model.apply_model(x, t, c, **kwargs)
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
            elif isinstance(c, dict):
                e_t = self.model.apply_model(x, t, c, **kwargs)
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
            else:
                raise NotImplementedError
            # text cfg
            if uc_type is None:
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            else:
                if uc_type == 'cfg_original':
                    e_t = e_t + unconditional_guidance_scale * (e_t - e_t_uncond)
                elif uc_type == 'cfg_ours':
                    e_t = e_t + unconditional_guidance_scale * (e_t_uncond - e_t)
                else:
                    raise NotImplementedError
            # temporal guidance
            if conditional_guidance_scale_temporal is not None:
                e_t_temporal = self.model.apply_model(x, t, c, **kwargs)
                e_t_image = self.model.apply_model(x, t, c, no_temporal_attn=True, **kwargs)
                e_t = e_t + conditional_guidance_scale_temporal * (e_t_temporal - e_t_image)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        
        if is_video:
            size = (b, 1, 1, 1, 1)
        else:
            size = (b, 1, 1, 1)
        a_t = torch.full(size, alphas[index], device=device)
        a_prev = torch.full(size, alphas_prev[index], device=device)
        sigma_t = torch.full(size, sigmas[index], device=device)
        sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        
        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        if self.use_scale:
            scale_arr = self.model.scale_arr if use_original_steps else self.ddim_scale_arr
            scale_t = torch.full(size, scale_arr[index], device=device)
            scale_arr_prev = self.model.scale_arr_prev if use_original_steps else self.ddim_scale_arr_prev
            scale_t_prev = torch.full(size, scale_arr_prev[index], device=device)
            pred_x0 /= scale_t 
            x_prev = a_prev.sqrt() * scale_t_prev * pred_x0 + dir_xt + noise
        else:
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0


    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)

        def extract_into_tensor(a, t, x_shape):
            b, *_ = t.shape
            out = a.gather(-1, t)
            return out.reshape(b, *((1,) * (len(x_shape) - 1)))

        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec

