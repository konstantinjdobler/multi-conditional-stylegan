# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from training.networks import normalize_2nd_moment
import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

#----------------------------------------------------------------------------


class Loss:
    # to be overridden by subclass
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        raise NotImplementedError()

#----------------------------------------------------------------------------


class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2,
                 training_set=None, condition_masking_p=0, condition_reconstruction_scaling=1):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.training_set = training_set
        self.condition_masking_p = condition_masking_p
        self.condition_reconstruction_scaling = torch.tensor(
            condition_reconstruction_scaling, device=device)
        self.once = False


    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        if not self.once:
            # Since the masking is stochastic, we need to intialize all metrics once for every process in order to avoid error down the line...
            training_stats.report(
                'Loss/G/reconstruction_loss', [])
            training_stats.report('Loss/G/reconstruction_mae', [])
            training_stats.report(
                'Loss/G/w_c_l2_dist', [])
            training_stats.report('Loss/G/w_c_orig_l2_dist', [])
            training_stats.report(
                'Loss/G/c_c_orig_l2_dist', [])
            self.once = True

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                # print(gen_c.shape)
                # Do condition masking
                if self.condition_reconstruction_scaling > 0:
                    original_gen_c = gen_c.detach().clone()
                    gen_c = self.training_set.dropout_mmdc_condition_batch(
                        gen_c, p=self.condition_masking_p)
                    masked_batch = not torch.equal(gen_c, original_gen_c)
                else:
                    masked_batch = False
                # May get synced by Gpl.
                gen_img, gen_ws = self.run_G(
                    gen_z, gen_c, sync=(sync and not do_Gpl))
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
            if masked_batch:
                masked_w = self.G_mapping(
                    gen_z, gen_c, skip_w_avg_update=True)[:, 0, :]
                unmasked_w = self.G_mapping(
                    gen_z, original_gen_c, skip_w_avg_update=True)[:, 0, :]
                reconstruction_loss = torch.nn.functional.mse_loss(
                    masked_w, unmasked_w)
                with torch.no_grad():
                    training_stats.report(
                        'Loss/G/reconstruction_loss', reconstruction_loss)
                    training_stats.report('Loss/G/reconstruction_mae', torch.nn.functional.l1_loss(
                    masked_w, unmasked_w))
                    embedded_masked_c = normalize_2nd_moment(
                        self.G_mapping.module.embed(gen_c).cpu())
                    embedded_unmasked_c = normalize_2nd_moment(
                        self.G_mapping.module.embed(original_gen_c).cpu())
                    training_stats.report(
                        'Loss/G/w_c_l2_dist', torch.nn.functional.mse_loss(masked_w.cpu(), embedded_masked_c))
                    training_stats.report('Loss/G/w_c_orig_l2_dist', torch.nn.functional.mse_loss(
                        masked_w.cpu(), embedded_unmasked_c))
                    training_stats.report(
                        'Loss/G/c_c_orig_l2_dist', torch.nn.functional.mse_loss(embedded_masked_c, embedded_unmasked_c))
             

            with torch.autograd.profiler.record_function('Gmain_backward'):
                total_loss = loss_Gmain.mean()
                if masked_batch:
                    total_loss += self.condition_reconstruction_scaling * reconstruction_loss
                total_loss.mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
