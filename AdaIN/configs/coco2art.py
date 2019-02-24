# Coco to Art 

resume = ''

train = dict(
     content_dataset = '../../../data/coco/images/train2017',
     style_dataset = '../../../data/wikiart/train',
     transform = dict(
        ),

     out = './results/coco2art',
     target_size = 256,
     crop_size = 512,

     iterations = 100000,
     batchsize = 16,

     save_interval = 10000,
     print_interval = 100,
     preview_interval = 100,

     parameters=dict(
         lr = 0.0001,
         beta1 = 0.,
         beta2 = 0.99,
         lam_c = 1,
         lam_s = 0.01,
     )
)
