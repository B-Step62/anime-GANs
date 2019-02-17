# dcgan 64x64

resume = ''

models = dict(
     model_type = 'dcgan',
     generator = dict(
         name = 'Generator128',
         z_dim = 128,
         top=512,
         norm = 'batch',
         ),
     discriminator = dict(
         name = 'Discriminator128',
         norm = None,
         top=64,
         use_sigmoid = False
         ),
)

train = dict(
     batchsize = 64,
     iterations = 1000000,
     dataset = '../../data/millionlive/face2/*/*',
     transform = dict(
        rotation = (-10, 10),
        ),

     out = './results/dcgan128-wgan-gp_face2',
     target_size = 128,

     loss_type = 'wgan-gp',

     save_interval = 100000,
     print_interval = 100,
     preview_interval = 2000,

     parameters=dict(
         g_lr = 0.0001,
         d_lr = 0.00005,
         lambda_gp = 10,
     )
)
