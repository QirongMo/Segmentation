
import os
import numpy as np
import argparse

import paddle
from dataloader import CelebA
from paddle.io import DataLoader
from visualdl import LogWriter

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--data_dir', type=str, default='./dataset/CelebAMask-HQ')
parser.add_argument('--image_list_file', type=str, default='./dataset/CelebAMask-HQ/train.txt')
parser.add_argument('--checkpoint_folder', type=str, default='./output')
parser.add_argument('--save_freq', type=int, default=1)
args = parser.parse_args()


def train(dataloader, model, criterion, optimzer):
    train_loss = 0.
    train_acc = 0.

    for batch_id, data in enumerate(dataloader):
        model.train()  # 切换到训练模式
        image = data[0].astype('float32').transpose((0, 3, 1, 2))
        label = data[1].astype('int64')
        pred = model(image)
        pred = paddle.nn.Softmax(axis=1)(pred)
        loss = criterion(pred.transpose((0, 2, 3, 1)), label)
        # 反向传播
        loss.backward()
        # 更新参数
        optimzer.step()
        # 梯度清零
        optimzer.clear_grad()

        n = image.shape[0]
        train_loss += loss.numpy()[0]

        # 计算accuracy
        pred = np.array(pred.numpy())
        pred = pred.argmax(axis=1)
        label = np.array(label.numpy()).reshape(label.shape[0], label.shape[1], label.shape[2])
        correct_prediction = np.equal(pred, label)
        accuracy = np.mean(correct_prediction)
        train_acc += accuracy
        print(f"\r  Step {batch_id + 1}/{200}，" +
              f"loss：{loss.numpy()[0]:.4f}，"+
              f"Average loss: {train_loss/(batch_id + 1):.4f}，" +
              f"accuracy：{accuracy:.4f}，"+
              f"Average accuracy: {train_acc/(batch_id + 1):.4f}"
              , end="")
        if batch_id + 1 == 200:
            break
    print()
    return train_loss/(batch_id + 1), train_acc/(batch_id + 1)


def main():
    train_dataset = CelebA(datadir=args.data_dir, img_size=512, mode='train')
    train_gen = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,)

    from model.PSPNet import PSPNet
    model = PSPNet(19)
    # 加载上一次训练的模型，继续训练
    param_dict = paddle.load('./output/PSPNet.pdparams')
    model.load_dict(param_dict)

    criterion = paddle.nn.CrossEntropyLoss(axis=3)
    scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=args.lr, factor=0.5, patience=3, verbose=True)
    # scheduler = paddle.optimizer.lr.StepDecay(learning_rate=args.lr, step_size=5, gamma=0.5, verbose=True)
    optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters(), weight_decay=1e-4)

    # 创建记录器
    log_writer = LogWriter(logdir=args.checkpoint_folder)

    for epoch in range(9, args.num_epochs + 1):
        print(f"Epoch {epoch}/{args.num_epochs}: ")
        train_loss, train_acc = train(train_gen, model, criterion, optimizer)
        scheduler.step(paddle.to_tensor(train_loss))
        log_writer.add_scalar(tag='loss', value=train_loss, step=epoch)
        log_writer.add_scalar(tag='acc', value=train_acc, step=epoch)

        if epoch % args.save_freq == 0 or epoch == args.num_epochs:
            # model_path = os.path.join(args.checkpoint_folder, f"Epoch-{epoch}-Loss-{train_loss:.4f}")
            model_path = os.path.join(args.checkpoint_folder, f"PSPNet-{epoch}.pdparams")
            model.eval()
            model_dict = model.state_dict()
            paddle.save(model_dict, model_path)

if __name__ == '__main__':
    main()