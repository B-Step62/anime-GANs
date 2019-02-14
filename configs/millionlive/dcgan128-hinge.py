# dcgan 64x64 hinge

resume = ''

models = dict(
     generator = dict(
         name = 'Generator128',
         z_dim = 64,
         norm = 'batch'
         ),
     discriminator = dict(
         name = 'Discriminator128',
         norm = '',
         use_sigmoid = False,
         ),
)

train = dict(
     batchsize = 64,
     iterations = 100000,
     dataset = '../data/face/*/*',
     out = './results/dcgan128-hinge',
     target_size = 128,

     loss_type = 'hinge',

     save_interval = 10000,
     print_interval = 100,
     preview_interval = 500,

     parameters=dict(
         g_lr = 0.001,
         d_lr = 0.0005,
     )
)
