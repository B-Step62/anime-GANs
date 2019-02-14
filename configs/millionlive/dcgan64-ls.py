# dcgan 64x64

resume = ''

models = dict(
     generator = dict(
         name = 'Generator64',
         z_dim = 64,
     ),
     discriminator = dict(
         name = 'Discriminator64',
     ),
)

train = dict(
     batchsize = 32,
     iterations = 100000,
     dataset = '../data/face/*/*',
     out = './results/dcgan64-ls',
     target_size = 64,

     loss_type = 'ls',

     save_interval = 5000,
     print_interval = 10,
     preview_interval = 100,

     parameters=dict(
         g_lr = 0.0001,
         d_lr = 0.00005,
     )
)
