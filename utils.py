import os
import torch
import torch.nn as nn
import copy
import pandas as pd
import matplotlib.pyplot as plt

def config(mode):
    choices_net = ['1', 'SRResNet', '2', 'SRGAN']
    model_to_train = input(f'Which model would you like to {mode}? Choices: [[1, SRResNet], [2, SRRGAN]]\n')
    while model_to_train not in choices_net:
        model_to_train = input('Invalid choice. Try again.\n')

    if model_to_train in ['1', 'SRResNet']:
        choices_lgen = ['1', 'MSE', '2', 'VGG22']
        choices_lgan = ['1', 'MSE', '2', 'VGG22', '3', 'VGG54']
        loss_function = input(f'Which loss are you using to {mode} this model? Choices: [[1, MSE], [2, VGG22]]\n')
        while loss_function not in choices_lgen:
            loss_function = input('Invalid choice. Try again.\n')
    elif model_to_train in ['2', 'SRGAN']:
        choices_lgan = ['1', 'MSE', '2', 'VGG22', '3', 'VGG54']
        loss_function = input(f'Which loss are you using to {mode} this model? Choices: [[1, MSE], [2, VGG22], [3, VGG54]]\n')
        while loss_function not in choices_lgan:
            loss_function = input('Invalid choice. Try again.\n')

    if model_to_train == choices_net[0]:
        model_to_train = choices_net[1]
    elif model_to_train == choices_net[2]:
        model_to_train = choices_net[3]

    if loss_function == choices_lgan[0]:
        loss_function = choices_lgan[1]
    elif loss_function == choices_lgan[2]:
        loss_function = choices_lgan[3]
    elif loss_function == choices_lgan[4]:
        loss_function = choices_lgan[5]

    return model_to_train, loss_function

def create_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def generate_folder_structure(model_name, loss_function):
    create_folder('results')
    create_folder(f'results/{model_name}')
    create_folder(f'results/{model_name}/{loss_function}')
    if model_name == 'SRGAN':
        create_folder(f'results/{model_name}/{loss_function}/models_gen')
        create_folder(f'results/{model_name}/{loss_function}/models_disc')
    else:
        create_folder(f'results/{model_name}/{loss_function}/models_gen')

def data_parallel(obj, device, ngpu):
    obj = obj.to(device)
    if device.type == 'cuda' and ngpu > 1:
        obj = nn.DataParallel(obj, list(range(ngpu)))
    return obj

def create_progress_dictionary(mode, in_loop):
    if mode == 'SRResNet':
        if in_loop:
            results = {
                'tLoss': [],
                'tPSNR': [],
                'vLoss': [],
                'vPSNR': []
            }
            return results
        else:
            results_by_epoch = {
                'tLoss': [],
                'tPSNR': [],
                'vLoss': [],
                'vPSNR': []
            }
            return results_by_epoch
    elif mode == 'SRGAN':
        if in_loop:
            results = {
                'tDiscLoss': [],
                'tDiscReal': [],
                'tDiscFakeBefore': [],
                'tDiscFakeAfter': [],
                'tGenLoss': [],
                'tPSNR': [],
                'vDiscLoss': [],
                'vDiscReal': [],
                'vDiscFake': [],
                'vGenLoss': [],
                'vPSNR': []
            }
            return results
        else:
            results_by_epoch = {
                'tGenLoss': [],
                'tDiscLoss': [],
                'tPSNR': [],
                'vGenLoss': [],
                'vDiscLoss': [],
                'vPSNR': []
            }
            return results_by_epoch

def append_progress_dictionary(model_name, results, results_by_epoch, batch_size, actual_batch_size):
    if model_name == 'SRResNet':
        num_iters = len(results['tLoss'])
        results_by_epoch['tLoss'].append(sum(results['tLoss']) / ((num_iters - 1) * batch_size + actual_batch_size))
        results_by_epoch['tPSNR'].append(sum(results['tPSNR']) / ((num_iters - 1) * batch_size + actual_batch_size))
        results_by_epoch['vLoss'].append(sum(results['vLoss']) / ((num_iters - 1) * batch_size + actual_batch_size))
        results_by_epoch['vPSNR'].append(sum(results['vPSNR']) / ((num_iters - 1) * batch_size + actual_batch_size))

    elif model_name == 'SRGAN':
        num_iters = len(results['tGenLoss'])
        results_by_epoch['tGenLoss'].append(sum(results['tGenLoss']) / ((num_iters - 1) * batch_size + actual_batch_size))
        results_by_epoch['tDiscLoss'].append(sum(results['tDiscLoss']) / ((num_iters - 1) * batch_size + actual_batch_size))
        results_by_epoch['tPSNR'].append(sum(results['tPSNR']) / ((num_iters - 1) * batch_size + actual_batch_size))
        results_by_epoch['vGenLoss'].append(sum(results['vGenLoss']) / ((num_iters - 1) * batch_size + actual_batch_size))
        results_by_epoch['vDiscLoss'].append(sum(results['vDiscLoss']) / ((num_iters - 1) * batch_size + actual_batch_size))
        results_by_epoch['vPSNR'].append(sum(results['vPSNR']) / ((num_iters - 1) * batch_size + actual_batch_size))

def val(perceptual_loss, SR, HR, batch_size):
    SR = rgb_to_ycbcr(SR, batch_size)
    HR = rgb_to_ycbcr(HR, batch_size)
    mse = perceptual_loss(SR, HR)
    psnr = 20 * (255 / mse.sqrt()).log10()
    return psnr

def rgb_to_ycbcr(img, batch_size):
    img = img * 0.5 + 1
    output = 16 + img[:, 0, :, :] * 65.481 + img[:, 1, :, :] * 128.553 + img[:, 2, :, :] * 24.966
    return output.view(batch_size, -1)

def progress_bar_SRResNet(results, loss, psnr, epoch, num_epochs, batch_size, actual_batch_size, bar, train):
    if train:
        results['tLoss'].append(loss.item() * actual_batch_size)
        results['tPSNR'].append(psnr.item() * actual_batch_size)

        epoch_progress = f'[{epoch:03d}/{num_epochs:03d}]'
        train_loss = f"{sum(results['tLoss']) / ((len(results['tLoss']) - 1) * batch_size + actual_batch_size):.6f}"
        train_psnr = f"{sum(results['tPSNR']) / ((len(results['tPSNR']) - 1) * batch_size + actual_batch_size):.6f}"
        bar.set_description(
            desc=f'{epoch_progress} Train Loss -> {train_loss} - Train PSNR -> {train_psnr}'
        )
    else:
        results['vLoss'].append(loss.item() * actual_batch_size)
        results['vPSNR'].append(psnr.item() * actual_batch_size)

        epoch_progress = f'[{epoch:03d}/{num_epochs:03d}]'
        val_loss = f"{sum(results['vLoss']) / ((len(results['vLoss']) - 1) * batch_size + actual_batch_size):.6f}"
        val_psnr = f"{sum(results['vPSNR']) / ((len(results['vPSNR']) - 1) * batch_size + actual_batch_size):.6f}"
        bar.set_description(
            desc=f'{epoch_progress} Val Loss -> {val_loss} - Val PSNR -> {val_psnr}'
        )

def progress_bar_SRGAN(results, loss_g, loss_d, d_output_real, d_output_fake, psnr, epoch, num_epochs, batch_size, actual_batch_size, bar, train):
    if train:
        results['tGenLoss'].append(loss_g.item() * actual_batch_size)
        results['tDiscLoss'].append(loss_d.item() * actual_batch_size)
        results['tDiscReal'].append(d_output_real.mean().item() * actual_batch_size)
        results['tDiscFakeAfter'].append(d_output_fake.mean().item() * actual_batch_size)                
        results['tPSNR'].append(psnr.item() * actual_batch_size)

        epochProgress = f'[{epoch:03d}/{num_epochs:03d}]'
        trainGenLoss = f"{sum(results['tGenLoss']) / ((len(results['tGenLoss']) - 1) * batch_size + actual_batch_size):.6f}"
        trainDiscLoss = f"{sum(results['tDiscLoss']) / ((len(results['tDiscLoss']) - 1) * batch_size + actual_batch_size):.6f}"
        trainPSNR = f"{sum(results['tPSNR']) / ((len(results['tPSNR']) - 1) * batch_size + actual_batch_size):.6f}"
        discriminatorOutputsReal = f"{sum(results['tDiscReal']) / ((len(results['tDiscReal']) - 1) * batch_size + actual_batch_size):.6f}"
        discriminatorOutputsFakeBefore = f"{sum(results['tDiscFakeBefore']) / ((len(results['tDiscFakeBefore']) - 1) * batch_size + actual_batch_size):.6f}"
        discriminatorOutputsFakeAfter = f"{sum(results['tDiscFakeAfter']) / ((len(results['tDiscFakeAfter']) - 1) * batch_size + actual_batch_size):.6f}"
        bar.set_description(
            desc=f"Train: {epochProgress} Gen Loss -> {trainGenLoss} - Disc Loss -> {trainDiscLoss} - D(HR) -> {discriminatorOutputsReal} - D(SR) -> {discriminatorOutputsFakeBefore}; {discriminatorOutputsFakeAfter} - PSNR -> {trainPSNR}"
        )
    else:
        results['vGenLoss'].append(loss_g.item() * actual_batch_size)
        results['vDiscLoss'].append(loss_d.item() * actual_batch_size)
        results['vDiscReal'].append(d_output_real.mean().item() * actual_batch_size)
        results['vDiscFake'].append(d_output_fake.mean().item() * actual_batch_size)                
        results['vPSNR'].append(psnr.item() * actual_batch_size)

        epochProgress = f'[{epoch:03d}/{num_epochs:03d}]'
        valGenLoss = f"{sum(results['vGenLoss']) / ((len(results['vGenLoss']) - 1) * batch_size + actual_batch_size):.6f}"
        valDiscLoss = f"{sum(results['vDiscLoss']) / ((len(results['vDiscLoss']) - 1) * batch_size + actual_batch_size):.6f}"
        valPSNR = f"{sum(results['vPSNR']) / ((len(results['vPSNR']) - 1) * batch_size + actual_batch_size):.6f}"
        discriminatorOutputsReal = f"{sum(results['vDiscReal']) / ((len(results['vDiscReal']) - 1) * batch_size + actual_batch_size):.6f}"
        discriminatorOutputsFake = f"{sum(results['vDiscFake']) / ((len(results['vDiscFake']) - 1) * batch_size + actual_batch_size):.6f}"
        bar.set_description(
            desc=f"Val: {epochProgress} Gen Loss -> {valGenLoss} - Disc Loss -> {valDiscLoss} - D(HR) -> {discriminatorOutputsReal} - D(SR) -> {discriminatorOutputsFake} - PSNR -> {valPSNR}"
        )

def save_model(model_name, loss_function, epoch, model_generator, optim_generator, loss_g, model_discriminator, optim_discriminator, loss_d, best_model):
    if epoch + 1 < 10:
        torch.save(
            {
                'epoch': epoch,
                'model_gen_state_dict': model_generator.state_dict(),
                'optimizer_gen_state_dict': optim_generator.state_dict(),
                'loss_g': loss_g
            }, f'results/{model_name}/{loss_function}/models_gen/gen_0{epoch + 1}.tar'
        )
        if model_name == 'SRGAN':
            torch.save(
                {
                    'epoch': epoch,
                    'model_disc_state_dict': model_discriminator.state_dict(), 
                    'optimizer_disc_state_dict': optim_discriminator.state_dict(),
                    'loss_d': loss_d
                },
                f'results/{model_name}/{loss_function}/models_disc/disc_0{epoch + 1}.tar'
            )
    else:
        torch.save(
            {
                'epoch': epoch,
                'model_gen_state_dict': model_generator.state_dict(),
                'optimizer_gen_state_dict': optim_generator.state_dict(),
                'loss_g': loss_g
            }, f'results/{model_name}/{loss_function}/models_gen/gen_{epoch + 1}.tar'
        )
        if model_name == 'SRGAN':
            torch.save(
                {
                    'epoch': epoch,
                    'model_disc_state_dict': model_discriminator.state_dict(), 
                    'optimizer_disc_state_dict': optim_discriminator.state_dict(),
                    'loss_d': loss_d
                },
                f'results/{model_name}/{loss_function}/models_disc/disc_{epoch + 1}.tar'
            )

    if best_model:
        torch.save(
            {
                'epoch': epoch,
                'model_gen_state_dict': model_generator.state_dict(),
                'optimizer_gen_state_dict': optim_generator.state_dict(),
                'loss_g': loss_g
            },
            f'results/{model_name}/{loss_function}/best_gen.tar'
        )
        if model_name == 'SRGAN':
            torch.save(
                {
                    'epoch': epoch,
                    'model_disc_state_dict': model_discriminator.state_dict(), 
                    'optimizer_disc_state_dict': optim_discriminator.state_dict(),
                    'loss_d': loss_d
                },
                f'results/{model_name}/{loss_function}/best_disc.tar'
            )

def generate_report(model_name, results_by_epoch, loss_function):
    if model_name == 'SRResNet':
        report = pd.DataFrame(
            data={
                'tLoss': results_by_epoch['tLoss'],
                'tPSNR': results_by_epoch['tPSNR'],
                'vLoss': results_by_epoch['vLoss'],
                'vPSNR': results_by_epoch['vPSNR']
            }
        )
        report.to_csv(f'results/{model_name}/{loss_function}/metrics.csv', index_label='Epoch', mode='w')

    elif model_name == 'SRGAN':
        report = pd.DataFrame(
            data={
                'tGenLoss': results_by_epoch['tGenLoss'],
                'tDiscLoss': results_by_epoch['tDiscLoss'],
                'tPSNR': results_by_epoch['tPSNR'],
                'vGenLoss': results_by_epoch['vGenLoss'],
                'vDiscLoss': results_by_epoch['vDiscLoss'],
                'vPSNR': results_by_epoch['vPSNR'],
            }
        )
        report.to_csv(f'results/{model_name}/{loss_function}/metrics.csv', index_label='Epoch', mode='w')

def show_image(tensor):
    plt.imshow((tensor * 0.5 + 0.5).permute(1, 2, 0).cpu().detach().numpy(), cmap='hsv')
    plt.show()

def load_gen(model_name, loss_function, model):
    checkpoint = torch.load(f'results/{model_name}/{loss_function}/best_gen.tar')
    model.load_state_dict(checkpoint['model_gen_state_dict'])
    return model