import torch
import torch.optim as optim
from torch.autograd import Variable


class Trainer:
    def __init__(self, args, model, criterion, logger):
        self.model = model
        self.criterion = criterion
        self.optimizer = optim.SGD(
            model.parameters(),
            args.learn_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
        self.nGPU = args.nGPU
        self.learn_rate = args.learn_rate
        self.architecture = args.model

    def train(self, epoch, train_loader):
        n_batches = len(train_loader)

        acc_avg = 0

        loss_avg = 0
        total = 0

        model = self.model
        model.train()
        self.learning_rate(epoch)

        for i, (input_tensor, target) in enumerate(train_loader):

            if self.nGPU > 0:
                input_tensor = input_tensor.cuda()
                target = target.cuda(async=True)

            batch_size = target.size(0)
            input_var = Variable(input_tensor)
            target_var = Variable(target)

            output = model(input_var)

            loss = self.criterion(output, target_var)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc = self.accuracy(output.data, target)

            acc_avg += acc * batch_size

            loss_avg += loss.data[0] * batch_size
            total += batch_size

            print("| Epoch[%d] [%d/%d]  Loss %1.4f  Acc %6.3f " % (
                epoch,
                i + 1,
                n_batches,
                loss.data[0],
                acc))

        loss_avg /= total
        acc_avg /= total

        print("\n=> Epoch[%d]  Loss: %1.4f  Acc %6.3f  \n" % (
            epoch,
            loss_avg,
            acc_avg))

        summary = dict()

        summary['acc'] = acc_avg
        summary['loss'] = loss_avg

        return summary

    def test(self, epoch, test_loader):
        n_batches = len(test_loader)

        acc_avg = 0

        total = 0

        model = self.model
        model.eval()

        for i, (input_tensor, target) in enumerate(test_loader):

            if self.nGPU > 0:
                input_tensor = input_tensor.cuda()
                target = target.cuda(async=True)

            batch_size = target.size(0)
            input_var = Variable(input_tensor)
            # target_var = Variable(target)

            output = model(input_var)

            acc = self.accuracy(output.data, target)

            acc_avg += acc * batch_size

            total += batch_size

            print("| Test[%d] [%d/%d]  Acc %6.3f  " % (
                epoch,
                i + 1,
                n_batches,
                acc))

        acc_avg /= total

        print("\n=> Test[%d]  Acc %6.3f\n" % (
            epoch,
            acc_avg))

        summary = dict()

        summary['acc'] = acc_avg

        return summary

    def accuracy(self, output, target):

        batch_size = target.size(0)

        _, pred = torch.max(output, 1)

        correct = pred.eq(target).float().sum(0)

        correct.mul_(100. / batch_size)

        return correct[0]

    def learning_rate(self, epoch):
        decay = 0.1 ** int(epoch / 5)
        learn_rate = self.learn_rate * decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learn_rate
