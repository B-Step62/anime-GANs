# sn_projection 128 hinge

resume = ''

models = dict(
     model_type = 'sn_projection',
     generator = dict(
         name = 'ResNetGenerator128',
         z_dim = 128,
         base=64,
         norm = 'c_batch',
         bottom_width = 4,
         ),
     discriminator = dict(
         name = 'SNResNetProjectionDiscriminator128',
         norm = 'spectrum',
         base=64,
         bottom_width = 4,
         ),
)

train = dict(
     batchsize = 32,
     iterations = 1000000,
     dataset_list = '/home/watanabe/M1/illustGAN/data/danbooru/face/more-1girl_hair_tag.txt',
     n_classes = 10,
     transform = dict(
        rotation = (-10, 10),
        ),

     out = './results/danbooru/sn_projection64-hinge',
     target_size = 64,

     loss_type = 'hinge',
     discriminator_iter = 5,

     save_interval = 20000,
     print_interval = 100,
     preview_interval = 1000,

     parameters=dict(
         g_lr = 0.0002,
         d_lr = 0.0002,
     )
)
