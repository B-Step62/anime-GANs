# resgan 128x128

resume = ''

models = dict(
        model_type = 'resgan',
        generator = dict(
         name = 'ResGenerator128',
         z_dim = 128,
         norm = 'batch',
         ),
     discriminator = dict(
         name = 'ResDiscriminator128',
         norm = None,
         use_sigmoid = False
         ),
)

train = dict(
     batchsize = 32,
     iterations = 1000000,
     dataset = '../data/face/*/*',
     out = './results/resgan128-wgan-gp',
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
