# dcgan 64x64

resume = ''

models = dict(
     model_type = 'dcgan',
     generator = dict(
         name = 'Generator128',
         z_dim = 64,
         norm = 'batch',
         ),
     discriminator = dict(
         name = 'Discriminator128',
         norm = None,
         use_sigmoid = False
         ),
)

train = dict(
     batchsize = 32,
     iterations = 1000000,
     dataset = '../data/danbooru/face/more-1girl',
     transform = dict(
        rotation = (-10, 10),
        ),

     out = './results/danbooru/dcgan128-wgan-gp',
     target_size = 128,

     loss_type = 'wgan-gp',

     save_interval = 100000,
     print_interval = 100,
     preview_interval = 1000,

     parameters=dict(
         g_lr = 0.0001,
         d_lr = 0.00005,
         lambda_gp = 10,
     )
)
