# pggan ls 256x256

resume = ''

models = dict(
     generator = dict(
         z_dim = 512,
         normalize_z = False,        #paper:False
         use_batchnorm = False,     #paper:False
         use_wscale = True,         #paper:True
         use_pixelnorm = True,      #paper:True
         tanh_at_end = False,        #paper:False
         activation = 'leaky_relu', #paper:leaky_relu
         ),
     discriminator = dict(
         initial_f_map = 8192,
         use_wscale = True,         #paper:True
         use_gdrop = True,         #paper:True
         use_layernorm = False,     #paper:False
         add_noise = False,         #paper:Fale
         ),
)

train = dict(
     dataset = '../../data/danbooru/face/more-1girl',
     transform = dict(
        #rotation = (-10, 10),
        ),

     out = './results/danbooru/pggan256-lsgan',
     target_size = 256,

     loss_type = 'ls',

     save_interval = 10000,
     print_interval = 100,
     preview_interval = 1000,

     # learning rate decay schedule
     rampup_kimg = 10000,
     rampdown_kimg = 10000,
     total_kimg = 10000,

     # stabilizing and fading network schedule
     stabilizing_kimg = 600,
     transition_kimg = 600,

     parameters=dict(
         g_lr = 0.001,
         d_lr = 0.001,
         beta1 = 0.,
         beta2 = 0.99,
         lambda_d_fake = 0.1,
     )
)
