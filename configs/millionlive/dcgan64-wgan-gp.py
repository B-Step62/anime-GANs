# dcgan 64x64 wga-gp

resume = ''

models = dict(
     generator = dict(
         name = 'Generator64',
         z_dim = 64,
         norm = 'batch',
     ),
     discriminator = dict(
         name = 'Discriminator64',
         norm = None,
         use_sigmoid = False,
     ),
)

train = dict(
     batchsize = 32,
     iterations = 1000000,
     dataset = '../data/face/*/*',
     out = './results/dcgan64-wgan-gp',
     target_size = 64,

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
