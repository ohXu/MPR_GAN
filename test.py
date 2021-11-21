import os
from collections import OrderedDict
from data.loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
from options import TestOptions
import time
import numpy as np

if __name__ == '__main__':

    opt = TestOptions().parse(save=False)
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    # test
    model = create_model(opt)
    Time = []
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break

        minibatch = 1
        start1 = time.time()
        generated, generatedHR = model.inference(data['image1'], data['image2'])
        Time.append(time.time() - start1)
        visuals = OrderedDict([('input_label', util.tensor2im(data['image1'][0])),
                               ('synthesized_image', util.tensor2im(generated.data[0])),
                               ('synthesized_image2', util.tensor2im(generatedHR.data[0]))
                               ])

        img_path = data['path1']
        print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)
    print(np.mean(Time))
    webpage.save()
