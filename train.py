import torch
import torch.optim as optim
from torch.autograd import Variable
import pickle
import torch.cuda


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
        acc_top10_avg = 0

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

            acc, acc_top10 = self.accuracy(output.data, target, (1, 10))

            acc_avg += acc * batch_size
            acc_top10_avg += acc_top10 * batch_size

            loss_avg += loss.data[0] * batch_size
            total += batch_size

            if i % 10 == 0:
                print("| Epoch[%d] [%d/%d]  Loss %1.4f  Acc %6.3f Acc-Top10 %6.3f" % (
                    epoch,
                    i + 1,
                    n_batches,
                    loss_avg / total,
                    acc_avg / total, acc_top10_avg / total))
                loss_avg = 0
                acc_avg = 0
                acc_top10_avg = 0
                total = 0

        loss_avg /= total
        acc_avg /= total

        print("\n=> Epoch[%d]  Loss: %1.4f  Acc %6.3f  \n" % (
            epoch,
            loss_avg,
            acc_avg))

        torch.cuda.empty_cache()

        summary = dict()

        summary['acc'] = acc_avg
        summary['loss'] = loss_avg

        return summary

    def test(self, epoch, test_loader):

        targets = []
        outputs = []

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

            acc, acc_top10 = self.accuracy(output.data, target, (1, 10))

            acc_avg += acc * batch_size

            total += batch_size

            print("| Test[%d] [%d/%d]  Acc %6.3f Acc-Top10 %6.3f" % (
                epoch,
                i + 1,
                n_batches,
                acc, acc_top10))

        acc_avg /= total

        print("\n=> Test[%d]  Acc %6.3f\n" % (
            epoch,
            acc_avg))

        torch.cuda.empty_cache()

        summary = dict()

        summary['acc'] = acc_avg

        return summary

    # def accuracy(self, output, target):
    #
    #     batch_size = target.size(0)
    #
    #     _, pred = torch.max(output, 1)
    #
    #     correct = pred.eq(target).float().sum(0)
    #
    #     correct.mul_(100. / batch_size)
    #
    #     return correct[0]

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            # print(correct_k)

            res.append(correct_k.mul_(100.0 / batch_size)[0])
        return res

    def learning_rate(self, epoch):
        decay = 0.1 ** ((epoch - 1) // 5)
        learn_rate = self.learn_rate * decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learn_rate
