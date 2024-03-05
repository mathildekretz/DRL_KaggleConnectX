import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm 
from utils import *
import os

from model import Connect4Model as nnet
cuda = torch.cuda.is_available()

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 5,
    'batch_size': 64,
    'num_channels': 64,
})

class AlphaZero():
    def __init__(self, model):
        self.model = model
    
    def train(self, boards, target_pis, target_vs, optimizer):
        self.model.train()

        out_log_pi, out_v = self.model(boards)

        pi_loss = -torch.sum(target_pis * out_log_pi) / target_pis.size()[0]
        v_loss = torch.sum((target_vs - out_v.view(-1))**2) / target_vs.size()[0]
        total_loss = pi_loss + v_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return total_loss, pi_loss, v_loss
    
    def predict(self, board):
        self.model.eval()

        with torch.no_grad():
            log_pi, v = self.model(board)
        
        pi = torch.exp(log_pi)
        return pi, v
        

class AlphaZeroAgent():
    def __init__(self, game):
        super(AlphaZeroAgent, self).__init__()
        self.model = nnet(game, args) #AlphaZero model 
        self.alg = AlphaZero(self.model)
        self.cuda = cuda
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
    
    def train(self, examples):
        optimizer = optim.Adam(self.alg.model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            print('EPOCH : ' + str(epoch+1))
            
            batch_count = int(len(examples) / args.batch_size)

            pbar = tqdm(range(batch_count), desc='Training Net')
            for _ in pbar:
                sample_ids = np.random.randint(
                    len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                if self.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(
                    ), target_pis.contiguous().cuda(), target_vs.contiguous(
                    ).cuda()

                total_loss, pi_loss, v_loss = self.alg.train(
                    boards, target_pis, target_vs, optimizer)

                pbar.set_postfix(Loss_pi=pi_loss.item(), Loss_v=v_loss.item())
    
    def predict(self, board):
        board = torch.FloatTensor(board.astype(np.float64))
        if self.cuda:
            board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)

        pi, v = self.alg.predict(board)

        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]


    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.model.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            print(f'No model in path {filepath}')
            raise (f'No model in path {filepath}')
        map_location = None if self.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.model.load_state_dict(checkpoint['state_dict'])