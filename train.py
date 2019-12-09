import torch
import torch.nn as nn
import torch.optim as optim
from data.utils import generate_partitions
from data.dataset_loaders import TrainDataloader, ValDataloader
import utils 
from tqdm import tqdm
from torchsummary import summary
import models.Generator as g

def train(model_name, loss_function, opt):
    ngpu = torch.cuda.device_count()
    device = torch.device('cuda' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
    best_loss = None
    gen_config = opt['Generator']['network_config']
    model_generator = g.Generator(
        input_nc=gen_config['input_nc'],
        output_nc=gen_config['output_nc'],
        nf=gen_config['nf'],
        num_resblocks=gen_config['num_resblocks'],
        upscale_factor=gen_config['upscale_factor'],
        norm_type=gen_config['norm_type'],
        act_type=gen_config['act_type'],
        init_weights=gen_config['init_weights']
    )
    model_generator = utils.data_parallel(model_generator, device, ngpu)   
    optim_generator = optim.Adam(
        model_generator.parameters(),
        lr=gen_config['optimizer_parameters']['lr']
    ) 
    if model_name == 'SRResNet':        
        num_epochs = opt['Generator']['train_parameters']['number_of_epochs']
        start_epoch = opt['Generator']['train_parameters']['start_epoch']            
        model_discriminator = None
        optim_discriminator = None
        adversarial_loss = None
        loss_d = None

    elif model_name == 'SRGAN':
        model_generator = utils.load_gen(model_generator)
        import models.Discriminator as d
        num_epochs = opt['Discriminator']['train_parameters']['number_of_epochs']
        start_epoch = opt['Discriminator']['train_parameters']['start_epoch']
        disc_config = opt['Discriminator']['network_config']
        model_discriminator = d.Discriminator(
            input_nc=disc_config['input_nc'],
            nf=disc_config['nf'],
            norm_type=disc_config['norm_type'],
            act_type=disc_config['act_type'],
            init_weights=disc_config['init_weights']
        )
        model_discriminator = utils.data_parallel(model_discriminator, device, ngpu)
        optim_discriminator = optim.Adam(
            model_discriminator.parameters(),
            lr=disc_config['optimizer_parameters']['lr']
        )
        optim_generator_scheduler = optim.lr_scheduler.StepLR(
            optim_generator,
            step_size=num_epochs // 2,
            gamma=opt['Generator']['network_config']['optimizer_parameters']['gamma']
        )
        optim_discriminator_scheduler = optim.lr_scheduler.StepLR(
            optim_discriminator,
            step_size=num_epochs // 2, 
            gamma=opt['Discriminator']['network_config']['optimizer_parameters']['gamma']
        )
        adversarial_loss = nn.BCELoss()        

    perceptual_loss = nn.MSELoss()

    if loss_function == 'VGG22':
        from models.VGG import VGG
        if model_name == 'SRResNet':
            from loss.TVLoss import TVLoss
            tv_loss = TVLoss()
        feat_config = opt['VGG']
        model_feat = VGG(
            layers=feat_config['layers'],
            bn=feat_config['bn'],
            loss_config=loss_function,
            pretrained=feat_config['pretrained']
        ).eval()
        model_feat = utils.data_parallel(model_feat, device, ngpu)
    elif loss_function == 'VGG54':
        from models.VGG import VGG
        feat_config = opt['VGG']
        model_feat = VGG(
            layers=feat_config['layers'],
            bn=feat_config['bn'],
            loss_config=loss_function,
            pretrained=feat_config['pretrained']
        ).eval()
        model_feat = utils.data_parallel(model_feat, device, ngpu)

    train_partition, val_partition = generate_partitions(opt['train_dataset_path'])

    train_loader = TrainDataloader(
        image_list=train_partition,
        crop_size=opt['Generator']['train_parameters']['crop_size'],
        upscale_factor=opt['Generator']['network_config']['upscale_factor']
    )

    val_loader = ValDataloader(
        image_list=val_partition,
        crop_size=opt['Generator']['train_parameters']['crop_size'],
        upscale_factor=opt['Generator']['network_config']['upscale_factor']
    )

    utils.generate_folder_structure(model_name, loss_function)
    results_by_epoch = utils.create_progress_dictionary(model_name, in_loop=False)
    
    for epoch in range(start_epoch, num_epochs):
        results = utils.create_progress_dictionary(model_name, in_loop=True)
        train_bar = tqdm(train_loader)
        model_generator.train()
        batch_size = opt['Generator']['train_parameters']['batch_size']
        if model_name == 'SRGAN':
            model_discriminator.train()            
            label_real = utils.data_parallel(torch.ones(batch_size), device, ngpu)
            label_fake = utils.data_parallel(torch.zeros(batch_size), device, ngpu)
        for LRimg, HRimg in train_bar:
            actual_batch_size = LRimg.size(0)
            if actual_batch_size != batch_size and model_name == 'SRGAN':
                label_real = label_real[:actual_batch_size]
                label_fake = label_fake[:actual_batch_size]
            LRimg = LRimg.to(device)
            HRimg = HRimg.to(device)
            if model_name == 'SRResNet':
                SRimg = model_generator.forward(LRimg)
                optim_generator.zero_grad()
                if loss_function != 'MSE':
                    SRfeat = model_feat.forward(SRimg)
                    HRfeat = model_feat.forward(HRimg)
                    loss_g = perceptual_loss(SRfeat, HRfeat) + tv_loss(SRimg, 2e-8)
                else:
                    loss_g = perceptual_loss(SRimg, HRimg)
                loss_g.backward()
                optim_generator.step()

                psnr = utils.val(perceptual_loss, SRimg, HRimg, actual_batch_size)
                utils.progress_bar_SRResNet(
                    results, 
                    loss_g, 
                    psnr, 
                    epoch, 
                    num_epochs, 
                    batch_size, 
                    actual_batch_size, 
                    train_bar,
                    train=True
                )
            elif model_name == 'SRGAN':
                d_output_real = model_discriminator.forward(HRimg)
                optim_discriminator.zero_grad()
                loss_d_real = adversarial_loss(d_output_real, label_real)
                loss_d_real.backward()

                SRimg = model_generator.forward(LRimg)
                d_output_fake_before = model_discriminator.forward(SRimg)
                loss_d_fake = adversarial_loss(d_output_fake_before, label_fake)
                loss_d_fake.backward()
                optim_discriminator.step()

                results['tDiscFakeBefore'].append(d_output_fake_before.mean().item() * actual_batch_size)

                loss_d = loss_d_real + loss_d_fake

                optim_generator.zero_grad()
                SRimg = model_generator.forward(LRimg)
                d_output_fake_after = model_discriminator.forward(SRimg)
                if loss_function != 'MSE':
                    SRfeat = model_feat.forward(SRimg)
                    HRfeat = model_feat.forward(HRimg)
                    loss_g = perceptual_loss(SRfeat, HRfeat) + adversarial_loss(d_output_fake_after, label_real)
                else:
                    loss_g = perceptual_loss(SRimg, HRimg) + adversarial_loss(d_output_fake_after, label_real)
                loss_g.backward()
                optim_generator.step()

                psnr = utils.val(perceptual_loss, SRimg, HRimg, actual_batch_size)
                utils.progress_bar_SRGAN(
                    results, 
                    loss_g, 
                    loss_d, 
                    d_output_real, 
                    d_output_fake_before,
                    d_output_fake_after, 
                    psnr, 
                    epoch, 
                    num_epochs, 
                    batch_size, 
                    actual_batch_size, 
                    train_bar,
                    train=True
                )
        val_bar = tqdm(val_loader)
        model_generator.eval()
        if model_name == 'SRGAN':
            model_discriminator.eval()
            label_real = utils.data_parallel(torch.ones(batch_size), device, ngpu)
            label_fake = utils.data_parallel(torch.zeros(batch_size), device, ngpu)
        for LRimg, HRimg in val_bar:
            actual_batch_size = LRimg.size(0)
            if actual_batch_size != batch_size and model_name == 'SRGAN':
                label_real = label_real[:actual_batch_size]
                label_fake = label_fake[:actual_batch_size]
            LRimg = LRimg.to(device)
            HRimg = HRimg.to(device)            
            if model_name == 'SRResNet':
                SRimg = model_generator.forward(LRimg)
                if loss_function != 'MSE':
                    SRfeat = model_feat.forward(SRimg)
                    HRfeat = model_feat.forward(HRimg)
                    loss = perceptual_loss(SRfeat, HRfeat) + tv_loss(SRfeat, 2e-8)
                else:
                    loss = perceptual_loss(SRimg, HRimg)
                psnr = utils.val(perceptual_loss, SRimg, HRimg, actual_batch_size)
                utils.progress_bar_SRResNet(
                    results, 
                    loss, 
                    psnr, 
                    epoch, 
                    num_epochs, 
                    batch_size, 
                    actual_batch_size, 
                    val_bar,
                    train=False
                )
            elif model_name == 'SRGAN':
                d_output_real = model_discriminator(HRimg)
                loss_d_real = adversarial_loss(d_output_real, label_real)

                SRimg = model_generator.forward(LRimg)
                d_output_fake = model_discriminator(SRimg)
                loss_d_fake = adversarial_loss(d_output_fake, label_fake)

                loss_d = loss_d_real + loss_d_fake

                if loss_function != 'MSE':
                    SRfeat = model_feat.forward(SRimg)
                    HRfeat = model_feat.forward(HRimg)
                    loss_g = perceptual_loss(SRfeat, HRfeat) + adversarial_loss(d_output_fake, label_real)
                else:
                    loss_g = perceptual_loss(SRimg, HRimg) + adversarial_loss(d_output_fake, label_real)

                psnr = utils.val(perceptual_loss, SRimg, HRimg, actual_batch_size)
                utils.progress_bar_SRGAN(
                    results, 
                    loss_g, 
                    loss_d, 
                    d_output_real, 
                    d_output_fake, 
                    None,
                    psnr, 
                    epoch, 
                    num_epochs, 
                    batch_size, 
                    actual_batch_size, 
                    val_bar,
                    train=False
                )

        if model_name == 'SRGAN':
            optim_discriminator_scheduler.step()
            optim_generator_scheduler.step()

        utils.append_progress_dictionary(
            model_name,
            results,
            results_by_epoch,
            batch_size,
            actual_batch_size
        )

        epoch_loss = results_by_epoch['vLoss'][-1]
        if best_loss is None or epoch_loss <= best_loss:
            best_model = True
            best_loss = epoch_loss

        utils.save_model(
            model_name,
            loss_function,
            epoch,
            model_generator,
            optim_generator,
            loss_g,
            model_discriminator,
            optim_discriminator,
            loss_d,
            best_model
        )

        utils.generate_report(model_name, results_by_epoch, loss_function)

        best_model = False
