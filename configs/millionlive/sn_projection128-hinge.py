# sn_projection 128 hinge

resume = ''

models = dict(
     model_type = 'sn_projection',
     generator = dict(
         name = 'ResNetGenerator128',
         z_dim = 128,
         base=64,
         norm = 'c_batch',
         bottom_width = 8,
         ),
     discriminator = dict(
         name = 'SNResNetProjectionDiscriminator128',
         norm = 'spectrum',
         base=64,
         bottom_width = 8,
         ),
)

train = dict(
     batchsize = 64,
     iterations = 1000000,
     dataset_list = ['../data/millionlive/face2/AkaneNonohara/*', '../data/millionlive/face2/AmiHutami/*', '../data/millionlive/face2/AnnaMochiduki/*', '../data/millionlive/face2/ArisaMatsuda/*', '../data/millionlive/face2/AyumuMaihama/*', '../data/millionlive/face2/AzusaMiura/*', '../data/millionlive/face2/ChihayaKisaragi/*', '../data/millionlive/face2/ChizuruNikaido/*', '../data/millionlive/face2/ElenaShimabara/*', '../data/millionlive/face2/Emily/*', '../data/millionlive/face2/FukaToyokawa/*', '../data/millionlive/face2/HarukaAmami/*', '../data/millionlive/face2/HibikiGanaha/*', '../data/millionlive/face2/HinataKinoshita/*', '../data/millionlive/face2/IkuNakatani/*', '../data/millionlive/face2/IoriMinase/*', '../data/millionlive/face2/Julia/*', '../data/millionlive/face2/KanaYabuki/*', '../data/millionlive/face2/KarenShinomiya/*', '../data/millionlive/face2/KonomiBaba/*', '../data/millionlive/face2/KotohaTanaka/*', '../data/millionlive/face2/Loco/*', '../data/millionlive/face2/MEgumiTokoro/*', '../data/millionlive/face2/MakotoKikuchi/*', '../data/millionlive/face2/MamiFutami/*', '../data/millionlive/face2/MatsuriTokugawa/*', '../data/millionlive/face2/MikiHoshi/*', '../data/millionlive/face2/MinakoSatake/*', '../data/millionlive/face2/MiraiKasuga/*', '../data/millionlive/face2/MiyaMiyao/*', '../data/millionlive/face2/MizukiMakabe/*', '../data/millionlive/face2/MomokoSuo/*', '../data/millionlive/face2/NaoYokoyama/*', '../data/millionlive/face2/NorikoFukuda/*', '../data/millionlive/face2/ReikaKitakami/*', '../data/millionlive/face2/RioMomose/*', '../data/millionlive/face2/RitsukoAkiduki/*', '../data/millionlive/face2/SayokoTakayama/*', '../data/millionlive/face2/SerikaHakozaki/*', '../data/millionlive/face2/ShihoKitazawa/*', '../data/millionlive/face2/ShizukaMogami/*', '../data/millionlive/face2/SubaruNagayoshi/*', '../data/millionlive/face2/TakamiOgami/*', '../data/millionlive/face2/TakaneSijo/*', '../data/millionlive/face2/TomokaTenkubashi/*', '../data/millionlive/face2/TsubasaIbuki/*', '../data/millionlive/face2/UmiKosaka/*', '../data/millionlive/face2/YayoiTakatsuki/*', '../data/millionlive/face2/YukihoHagiwara/*', '../data/millionlive/face2/YurikoNanao/*'],
     n_classes = 50,
     transform = dict(
        rotation = (-10, 10),
        ),

     out = './results/sn_projection128-hinge_face2',
     target_size = 128,

     loss_type = 'hinge',
     discriminator_iter = 5,

     save_interval = 100000,
     print_interval = 100,
     preview_interval = 1000,

     parameters=dict(
         g_lr = 0.0002,
         d_lr = 0.0001,
     )
)
