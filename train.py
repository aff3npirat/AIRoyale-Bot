import os
import copy

import torch

import play
from bots.single_deck.nn import QNet
from bots.single_deck.bot import NEXT_CARD_END, SingleDeckBot



class Memory:

    def __init__(self, size, alpha, beta, eps):
        self.size = size
        self.data = [None for _ in range(size)]
        self.priorities = torch.zeros(self.size)
        self.new_data = []
        self.data_index = 0
        self.full = False
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def sample(self, num_samples):
        probs = self.priorities / torch.sum(self.priorities)

        if len(self.new_data) > 0:
            idxs = self.new_data[:num_samples]
            del self.new_data[:num_samples]
            
            weights = torch.ones(num_samples)
            
            if len(idxs) < num_samples:
                idxs_ = torch.multinomial(probs, num_samples=num_samples-len(idxs), replacement=False)
                idxs.extend(idxs_.tolist())
                weights[len(idxs):] = (1/(probs[idxs_]*self.size))**self.beta
        else:
            idxs = torch.multinomial(probs, num_samples=num_samples, replacement=False).tolist()
            weights = (1/(probs[idxs]*self.size))**self.beta

        batch = []
        for i in idxs:
            batch.append(self.data[i])

        return batch, idxs, weights

    def _safe_add(self, entries):
        new_index = len(entries) + self.data_index

        self.new_data = [i for i in range(self.data_index, new_index)] + self.new_data

        self.data[self.data_index:new_index] = entries
        self.data_index = new_index if new_index < self.size else 0

    def add(self, entries):
        if len(entries) > self.size:
            entries = entries[-self.size:]

        new_index = len(entries) + self.data_index
        if new_index <= self.size:
            self._safe_add(entries)
        else:
            self.full = True
            overflow = new_index - self.size
            self._safe_add(entries[:-overflow])
            self._safe_add(entries[-overflow:])

    def update(self, indices, errors):
        priorities = errors**self.alpha + self.eps
        self.priorities[indices] = priorities

    def is_full(self):
        return self.full
    
    def __len__(self):
        if self.full:
            return self.size
        else:
            return self.data_index
        
    def __getitem__(self, index):
        if not (0 <= index < self.__len__()):
            raise IndexError(f"Index {index} is out of bounds for size {self.size}")

        return self.data[index]

def n_step_return(n, memory, start_idx, discount):
    reward = 0
    for i in range(n):
        idx = (start_idx + i) % len(memory)
        _, _, r_t, done = memory[idx]
        reward += r_t * discount**i

        if done:
            break

    return reward, idx


class Trainer:

    def __init__(self, output, num_games, hparams, ports, checkpoint=None, device="cpu"):
        os.makedirs(output, exist_ok=True)

        self.device = torch.device(device)

        self.output = output
        self.ports = ports
        self.output = output
        self.weight_decay = hparams["weight_decay"]
        self.batch_size = hparams["batch_size"]
        self.discount = hparams["discount"]
        self.delta = hparams["delta"]
        self.n = hparams["n"]
        self.num_games = num_games
        self.deck_names = hparams["deck_names"]
        self.unit_model = "./models/units_singledeck_cpu.onnx"
        self.side_model = "./models/side_cpu.onnx"
        self.number_model = "./models/number_cpu.onnx"

        if checkpoint is None:
            self.eps = hparams["eps"]
            lr = hparams["lr"]
            self.game_count = 0
            self.update_count = 0
            self.delta_count = 0
            self.memory = Memory()
        else:
            cp = torch.load(checkpoint)

            self.memory = cp["memory"]

            self.eps = cp["eps"]
            lr = cp["lr"]

            main_net_state_dict = cp["main_net"]    
            target_net_state_dict = cp["target_net"]

            self.game_count = cp["game_counter"]
            self.update_count = cp["update_counter"]
            self.delta_count = cp["delta_count"]

        self.main_net = QNet([512+NEXT_CARD_END, 128, 64, 5], activation="sigmoid", bias=True, feature_extractor=False)
        if checkpoint is not None:
            self.main_net.load_state_dict(main_net_state_dict)
        self.main_net.train()

        self.target_net = copy.deepcopy(self.main_net)
        if checkpoint is not None:
            self.target_net.load_state_dict(target_net_state_dict)
        self.target_net.eval()

        self.optim = torch.optim.SGD(self.main_net.parameters(), lr=lr, weight_decay=self.weight_decay)

    def checkpoint(self, name):
        path = os.path.join(self.output, name)

        cp = {
            "main_net": self.main_net.cpu().state_dict(),
            "target_net": self.target_net.cpu().state_dict(),
            "memory": self.memory,
            "eps": self.eps,
            "lr": self.optim.param_groups[0]["lr"],
            "game_counter": self.game_count,
            "update_counter": self.update_count,
            "delta_count": self.delta_count,
        }

        torch.save(cp, path)

    def update_target_net(self):
        self.target_net = copy.deepcopy(self.main_net)
        self.target_net.eval()

    def train(self, batch_size, num_batches, device):
        for b in range(num_batches):
            batch, idxs, is_weights = self.memory.sample(batch_size)  # list of tuples ((board, context), action, reward, done)

            board = torch.empty((batch_size, *batch[0][0][0].shape))
            context = torch.empty((batch_size, *batch[0][0][1].shape))
            action = torch.empty(batch_size)
            n_step_board = torch.empty((batch_size, *batch[0][0][0].shape))
            n_step_context = torch.empty((batch_size, *batch[0][0][1].shape))
            dones = torch.empty(batch_size)
            discounted_rewards = torch.empty(batch_size)
            actual_n = torch.empty(batch_size)
            for i in range(batch_size):
                discounted_reward, last_idx = n_step_return(self.n, self.memory, idxs[i], self.discount)
                discounted_rewards[i] = discounted_reward
                
                if self.memory[last_idx][3]:  # done
                    dones[i] = 1.0
                else:
                    state_n_step = self.memory[last_idx+1%len(self.memory)][0]
                    
                    n_step_board[i] = state_n_step[0]
                    n_step_context[i] = state_n_step[1]
                    dones[i] = 0.0
                    

            discounted_rewards = discounted_rewards.to(device)
            n_step_board = n_step_board.to(device)
            n_step_context = n_step_context.to(device)
            board = board.to(device)
            context = context.to(device)    
            with torch.no_grad():
                mask = (dones==0)
                state_n_step = (n_step_board[mask], n_step_context[mask])
                action_n_step = self.main_net(state_n_step)
                action_n_step[SingleDeckBot.get_illegal_actions(state_n_step)]
                action_n_step = torch.argmax(action_n_step, dim=1)
                q_n_step = self.target_net(state_n_step)[action_n_step]
                discounted_rewards[mask] += q_n_step*torch.pow(self.discount, actual_n)

            predicted_q = self.main_net((board, context))
            target_q = predicted_q.detach().clone()
            target_q[action] = discounted_rewards

            abs_errors = torch.sum(torch.abs(target_q - predicted_q), dim=1)

            loss = torch.mean((abs_errors**2) * is_weights.to(device))
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.memory.update(idxs, abs_errors.detach().cpu())

            self.delta_count += 1
            self.update_count += len(batch)

            if self.delta_count%self.delta == 0:
                self.update_target_net()
    
    def run(self):
        self.main_net = self.main_net.to(self.device)
        self.target_net = self.target_net.to(self.device)

        for _ in range(self.num_games):
            episodes = play.run(
                n_games=1,
                output=self.output,
                deck_names=self.deck_names,
                ports=self.ports,
                unit_model=self.unit_model,
                side_model=self.side_model,
                number_model=self.number_model,
                eps=self.eps,
                network=self.main_net.cpu().state_dict(),
            )

            self.game_count += 1
            self.memory.add(episodes)

            num_batches = len(episodes)//self.batch_size
            if len(episodes)%self.batch_size != 0:
                num_batches += 1

            self.train(batch_size=self.batch_size, num_batches=num_batches, device=self.device)  # train on new experience
            if self.memory.is_full():
                self.train(batch_size=self.batch_size, num_batches=1, device=self.device)  # train on random experience

            self.checkpoint("last.pt")






