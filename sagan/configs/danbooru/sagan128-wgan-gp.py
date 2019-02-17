# sagan 64x64

resume = ''

models = dict(
     generator = dict(
         name = 'ResNetGenerator128',
         norm = 'batch',
         z_dim = 128,
         ),
     discriminator = dict(
         name = 'ResNetProjectionDiscriminator128',
         norm = None,
         ),
)

train = dict(
     batchsize = 32,
     iterations = 1000000,
     dataset = '../../data/danbooru/face/more-1girl',
     transform = dict(
        #rotation = (-10, 10),
        ),

     out = './results/danbooru/sagan128-lsgan',
     target_size = 128,

     loss_type = 'wgan-gp',

     save_interval = 100000,
     print_interval = 100,
     preview_interval = 1000,

     discriminator_iter = 1,
     parameters=dict(
         g_lr = 0.000025,
         d_lr = 0.0001,
         adam_beta1 = 0.0,
         adam_beta2 = 0.9,
         lambda_gp = 10,
     )
)
