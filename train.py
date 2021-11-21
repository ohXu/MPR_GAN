# -*- coding:utf-8 -*-
import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from options import TrainOptions, TestOptions
from data.loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import util.util as util
import cv2

if __name__ == '__main__':

    opt = TrainOptions().parse()
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    # opt.continue_train = True
    if opt.continue_train:
        try:
            start_epoch, _ = np.loadtxt(iter_path, delimiter=',', dtype=int)
        except:
            start_epoch = 1
        print('Resuming from epoch %d' % (start_epoch))
    else:
        start_epoch = 1

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    opt2 = TestOptions().parse(save=False)
    opt2.nThreads = 1
    opt2.batchSize = 1
    opt2.serial_batches = True
    opt2.no_flip = True
    data_loader2 = CreateDataLoader(opt2)
    dataset2 = data_loader2.load_data()

    model = create_model(opt)
    visualizer = Visualizer(opt)
    optimizer_G, optimizer_D, optimizer_D2 = model.module.optimizer_G, model.module.optimizer_D, model.module.optimizer_D2

    f = open('loss.txt', 'w')
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):

        epoch_start_time = time.time()
        for i, data in enumerate(dataset, start=0):

            ############## Forward Pass ######################
            generated, generated_HR, losses = model(0, Variable(data['image1']), Variable(data['image2']))

            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.module.loss_names, losses))
            loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['G_SSIM']
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_D2 = (loss_dict['D_fake2'] + loss_dict['D_real2']) * 0.5

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # optimizer_D.zero_grad()
            # loss_D.backward()
            # optimizer_D.step()

            optimizer_D2.zero_grad()
            loss_D2.backward()
            optimizer_D2.step()

            ############## Display results and errors ##########
            if i % opt.print_freq == 0:
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                visualizer.print_current_errors(epoch, i, errors)
                # print(data['path1'][0], data['path2'][0])
                visuals = OrderedDict([('input_label', util.tensor2im(data['image1'][0])),
                                       ('synthesized_image', util.tensor2im(generated.data[0])),
                                       ('synthesized_image2', util.tensor2im(generated_HR.data[0])),
                                       ('real_image', util.tensor2im(data['image2'][0]))])
                visualizer.display_current_results(visuals, epoch)

        iter_end_time = time.time()
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (epoch))
            model.module.save('latest')
            model.module.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')
            model2 = create_model(opt2)
            s1 = 0
            s2 = 0
            s3 = 0
            nu = 0
            for i, data in enumerate(dataset2):

                a = int(data['path1'][0].split('\\')[-1].split('.')[0])
                if a % 5 == 0:
                    nu = nu + 1
                    generated, generatedHR = model2.inference(data['image1'], data['image2'])
                    visuals = OrderedDict([('input_label', util.tensor2im(data['image1'][0])),
                                           ('synthesized_image', util.tensor2im(generated.data[0])),
                                           ('synthesized_image2', util.tensor2im(generatedHR.data[0])),
                                           ('real_image', util.tensor2im(data['image2'][0]))
                                           ])

                    img_path = data['path1']
                    print('process image... %s' % img_path)
                    for label, image_numpy in visuals.items():
                        image_name = '%s_%s.jpg' % (str(a), label)
                        save_path = os.path.join('./vel/', image_name)
                        util.save_image(image_numpy, save_path)
                    target = cv2.imread('./vel/' + str(a) + '_' + 'synthesized_image2.jpg')
                    ref = cv2.imread('./vel/' + str(a) + '_' + 'real_image.jpg')
                    s = util.ssim(target, ref)
                    s1 += s[0]
                    s2 += s[1]
                    s = util.psnr(target, ref)
                    s3 += s
            s1 = s1 / nu
            s2 = s2 / nu
            s3 = s3 / nu
            print(nu, s1, s2, s3)
            f.write(str(epoch) + ' ' + str(s1) + ' ' + str(s2) + ' ' + str(s3) + '\n')

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            model.module.update_learning_rate()
    f.close()
