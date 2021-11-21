import numpy as np
import torch
import os
from torch.autograd import Variable
from .base_model import BaseModel
from . import networks


def create_model(opt):
    if opt.isTrain:
        model = MPGGANModel()
    else:
        model = InferenceModel()
    model.initialize(opt)
    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model


class MPGGANModel(BaseModel):

    def init_loss_filter(self):
        flags = (True, True, True, True, True, True, True)

        def loss_filter(g_gan, g_gan_feat, g_ssim, d_real, d_fake, d_real2, d_fake2):
            return [l for (l, f) in zip((g_gan, g_gan_feat, g_ssim, d_real, d_fake, d_real2, d_fake2), flags) if f]

        return loss_filter

    def initialize(self, opt):

        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        input_nc = opt.input_nc

        ##### define networks
        # Generator network
        netG_input_nc = input_nc
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.norm, gpu_ids=self.gpu_ids)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc

            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                          opt.num_D, gpu_ids=self.gpu_ids, ord='first')

            self.netD_2 = networks.define_D(6, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                            opt.num_D, gpu_ids=self.gpu_ids, ord='second')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter()
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            # self.criterionFeat = torch.nn.SmoothL1Loss()
            # nn.MSELoss()  L1Loss
            self.criterionFeat = torch.nn.SmoothL1Loss()
            self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_SSIM', 'D_real', 'D_fake', 'D_real2',
                                               'D_fake2')

            params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            params = list(self.netD_2.parameters())
            self.optimizer_D2 = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, real_image=None, real_image2=None, infer=False):

        real_image = Variable(real_image.data.cuda(), volatile=infer)
        real_image2 = Variable(real_image2.data.cuda())

        return real_image, real_image2

    def forward(self, ord, image1, image2, infer=False):

        real_image, real_image2 = self.encode_input(image1, image2)
        if ord == 0:
            # loss_G_GAN2 = loss_G_VGG2 = loss_D_real2 = loss_G_GAN_Feat2 = loss_D_fake2 = loss_G_VGG = 0
            # fake_image = self.netG.forward(input_label, input_label2)
            # pred_fake = self.netD.forward(torch.cat((input_label2, fake_image), dim=1))
            # loss_G_GAN = self.criterionGAN(pred_fake, True)
            # loss_G_GAN_Feat = (1 - networks.ssim(fake_image, real_image2)) * 5.0 + self.criterionFeat(fake_image,
            #                                                                                          real_image2) * 5.0
            #
            # # loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
            # # loss_G_VGG2 = self.criterionVGG(fake_image_HR, real_image2) * self.opt.lambda_feat
            #
            # input_concat = torch.cat((input_label2, fake_image.detach()), dim=1)
            # pred_fake_pool = self.netD.forward(input_concat)
            # loss_D_fake = self.criterionGAN(pred_fake_pool, False)
            #
            # input_concat1 = torch.cat((input_label2, real_image2), dim=1)
            # pred_real = self.netD.forward(input_concat1)
            # loss_D_real = self.criterionGAN(pred_real, True)
            #
            # if not self.opt.no_ganFeat_loss:
            #     feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            #     D_weights = 1.0 / self.opt.num_D
            #     for i in range(self.opt.num_D):
            #         for j in range(len(pred_fake[i]) - 1):
            #             loss_G_GAN_Feat += D_weights * feat_weights * \
            #                                self.criterionFeat(pred_fake[i][j],
            #                                                   pred_real[i][j].detach()) * self.opt.lambda_feat
            #
            # return [fake_image, fake_image,
            #         self.loss_filter(loss_G_GAN + loss_G_GAN2, loss_G_GAN_Feat + loss_G_GAN_Feat2,
            #                          loss_G_VGG + loss_G_VGG2, loss_D_real, loss_D_fake, loss_D_real2, loss_D_fake2)]
            loss_G_GAN = 0
            loss_G_GAN_Feat = loss_G_GAN_Feat2 = 0
            loss_D_real = loss_D_fake = 0

            fake_image, fake_image_HR = self.netG.forward(real_image)
            # pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))
            # loss_G_GAN = self.criterionGAN(pred_fake, True)
            pred_fake2 = self.netD_2.forward(fake_image_HR)
            loss_G_GAN2 = self.criterionGAN(pred_fake2, True)

            # loss_G_ssim = (1 - networks.ssim(fake_image, real_image)) * 10.0
            loss_G_ssim2 = (1 - networks.ssim(fake_image_HR, real_image2)) * 10.0

            # input_concat = torch.cat((input_label, fake_image.detach()), dim=1)
            # pred_fake_pool = self.netD.forward(input_concat)
            # loss_D_fake = self.criterionGAN(pred_fake_pool, False)
            # input_concat1 = torch.cat((input_label, real_image), dim=1)
            # pred_real = self.netD.forward(input_concat1)
            # loss_D_real = self.criterionGAN(pred_real, True)

            pred_fake2_pool = self.netD_2.forward(fake_image_HR.detach())
            pred_real2 = self.netD_2.forward(real_image2)
            loss_D_fake2 = self.criterionGAN(pred_fake2_pool, False)
            loss_D_real2 = self.criterionGAN(pred_real2, True)

            # if not self.opt.no_ganFeat_loss:
            #     feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            #     D_weights = 1.0 / self.opt.num_D
            #     for i in range(self.opt.num_D):
            #         for j in range(len(pred_fake[i]) - 1):
            #             loss_G_GAN_Feat += D_weights * feat_weights * \
            #                                self.criterionFeat(pred_fake[i][j],
            #                                                   pred_real[i][j].detach()) * self.opt.lambda_feat

            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake2[i]) - 1):
                    loss_G_GAN_Feat2 += D_weights * feat_weights * \
                                        self.criterionFeat(pred_fake2[i][j],
                                                           pred_real2[i][j].detach()) * self.opt.lambda_feat

            return [fake_image, fake_image_HR,
                    self.loss_filter(loss_G_GAN + loss_G_GAN2, loss_G_GAN_Feat + loss_G_GAN_Feat2, loss_G_ssim2,
                                     loss_D_real, loss_D_fake, loss_D_real2, loss_D_fake2)]

        # elif ord == 1:
        #     # fake_image, fake_image_HR = self.netG.forward(input_concat)
        #     # real_out = self.netD_2.forward(real_image2).mean()
        #     # fake_out = self.netD_2.forward(fake_image_HR).mean()
        #     # loss_D2 = 1 - real_out + fake_out
        #     # return [loss_D2, fake_image, fake_image_HR]
        #
        #     fake_image, fake_image_HR = self.netG.forward(input_concat)
        #     pred_fake = self.netD_2.forward(fake_image_HR)
        #     pred_real = self.netD_2.forward(real_image2)
        #     loss_D_fake = self.criterionGAN(pred_fake, False)
        #     loss_D_real = self.criterionGAN(pred_real, True)
        #     loss_D = (loss_D_fake + loss_D_real)*0.5
        #
        #     return [loss_D, fake_image, fake_image_HR]

    def inference(self, image=None, image2=None):

        input_label, real_image2 = self.encode_input(image, image2, infer=True)
        fake_image, fake_image_HR = self.netG.forward(input_label)

        return fake_image, fake_image_HR

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


class InferenceModel(MPGGANModel):
    def forward(self, inp):
        image1, image2 = inp
        return self.inference(image1, image2)
