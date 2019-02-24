# pggan wgan-gp 256x256

resume = ''

models = dict(
     generator = dict(
         z_dim = 512,
         normalize_z = False,
         use_batchnorm = False,
         use_wscale = True,
         use_pixelnorm = True,
         tanh_at_end = False,
         activation = 'leaky_relu',
         ),
     discriminator = dict(
         initial_f_map = 8192,
         use_wscale = True,
         use_gdrop = True,
         use_layernorm = False,
         add_noise = False,
         ),
)

train = dict(
     dataset = '../../data/danbooru/face/more-1girl',
     transform = dict(
        #rotation = (-10, 10),
        ),

     out = './results/danbooru/pggan256_wscaled_-wgan-gp',
     target_size = 256,

     loss_type = 'wgan-gp',

     save_interval = 100000,
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
         lambda_gp = 10,
         lambda_d_fake = 1.0,
     )
)
