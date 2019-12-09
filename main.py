from config.options import parse
import utils
from train import train
from torch.backends import cudnn

### Loading options from JSON file into [opt] variable
train_file = 'config/train.json'
opt = parse(train_file)

cudnn.benchmark = opt['determinism']['benchmark_algorithms']
cudnn.deterministic = opt['determinism']['state']

model_name, loss_name = utils.config('train')

if __name__ == '__main__':
    train(model_name, loss_name, opt)