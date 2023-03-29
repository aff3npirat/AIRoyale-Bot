import os
import copy
import time

import torch
import h5py

import play
from bots.single_deck.nn import QNet
from bots.single_deck.bot import NEXT_CARD_END, SingleDeckBot



class DiskMemory:

    def __init__(self, file, max_size=2**16, shape_dict=None, dtype_dict=None):
        try:
            with h5py.File(file, "r") as fout:
                pass
        except FileNotFoundError:
            with h5py.File(file, "w-") as fout:
                fout.attrs["groups"] = 0
                fout.attrs["group_size"] = max_size
                fout.attrs["size"] = 0
                fout.attrs["names"] = list(shape_dict.keys())
                for key in shape_dict:
                    fout.attrs[f"{key}_shape"] = shape_dict[key]
                    fout.attrs[f"{key}_dtype"] = dtype_dict[key]

                DiskMemory._add_group(fout)

        self.file = file

    def add(self, entries):
        N = len(entries)

        with h5py.File(self.file, "r+") as fout:
            max_size = fout.attrs["group_size"]

            group_id = fout.attrs["groups"]
            group = fout[f"{group_id}"]

            data_pointer = group.attrs["data_pointer"]
            idx = data_pointer

            for i in range(N):
                exp = SingleDeckBot.exp_to_dict(entries[i])
                for key in fout.attrs["names"]:
                    group[key][idx+1] = exp[key]

                idx += 1

                if idx == max_size-1:
                    group.attrs["data_pointer"] = max_size - 1
                    fout.attrs["size"] += idx - data_pointer
                    group = DiskMemory._add_group(fout)

                    data_pointer = group.attrs["data_pointer"]
                    idx = data_pointer

            group.attrs["data_pointer"] = idx
            fout.attrs["size"] += idx - data_pointer

    @staticmethod
    def _add_group(file):
        new_group = file.attrs["groups"] + 1
        max_size = file.attrs["group_size"]

        group = file.create_group(f"{new_group}")
        group.attrs["data_pointer"] = -1
        for name in file.attrs["names"]:
            shape = file.attrs[f"{name}_shape"]
            dtype = file.attrs[f"{name}_dtype"]
            group.create_dataset(name, shape=(max_size, *shape), dtype=dtype, compression="gzip", compression_opts=6)

        file.attrs["groups"] = new_group

        return group


class Memory:

    def __init__(self, size, alpha, beta, eps, beta_decay):
        self.size = size
        self.data = [None for _ in range(size)]
        self.priorities = torch.zeros(self.size)
        self.new_data = []
        self.data_index = 0
        self.full = False
        self.alpha = alpha
        self.beta = beta
        self.beta_decay = beta_decay
        self.eps = eps

    def sample(self, num_samples, shuffle=True):
        probs = self.priorities / torch.sum(self.priorities)

        if len(self.new_data) > 0:
            idxs = self.new_data[:num_samples]
            del self.new_data[:num_samples]
            
            weights = torch.ones(len(idxs))
        else:
            idxs = torch.multinomial(probs, num_samples=num_samples, replacement=False).tolist()
            weights = (1/(probs[idxs]*self.size))**self.beta
            self.beta *= self.beta_decay

        if shuffle:
            shuffled_idx = torch.randperm(num_samples)
            weights = weights[shuffled_idx]
            idxs = torch.tensor(idxs)[shuffled_idx].tolist()

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

    def __init__(self, hparams, ports, options, checkpoint=None, log_fn=None):
        self.device = torch.device(options["device"])
        self.cp_freq = options["checkpoint_frequency"]  # number of games
        self.output = options["output"]
        self.disk_memory = options["disk_memory"]

        if log_fn is None:
            self.logger = lambda x: None
        else:
            self.logger = log_fn

        self.ports = ports
        self.weight_decay = hparams["weight_decay"]
        self.batch_size = hparams["batch_size"]
        self.discount = hparams["discount"]
        self.delta = hparams["delta"]
        self.eps_decay = hparams["eps_decay"]
        self.n = hparams["n"]
        self.deck_names = hparams["deck_names"]
        self.unit_model = "./models/units_singledeck_cpu.onnx"
        self.side_model = "./models/side_cpu.onnx"
        self.number_model = "./models/number_cpu.onnx"

        if checkpoint is None:
            self.eps = hparams["eps0"]
            lr = hparams["lr0"]
            self.game_count = 0
            self.update_count = 0
            self.delta_count = 0
            self.memory = Memory(hparams["mem_size"], hparams["alpha0"], hparams["beta0"], hparams["min_sample_prob"], hparams["beta_decay"])
            self.time_elapsed = 0
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
            self.time_elapsed = cp["training_time"]

            self.logger(f"Loaded checkpoint '{checkpoint}'")

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
        if not os.path.exists(self.output):
            os.makedirs(self.output, exist_ok=True)

        path = os.path.join(self.output, name)

        self.time_elapsed += time.time() - self.tic
        self.tic = time.time()

        cp = {
            "main_net": self.main_net.cpu().state_dict(),
            "target_net": self.target_net.cpu().state_dict(),
            "memory": self.memory,
            "eps": self.eps,
            "lr": self.optim.param_groups[0]["lr"],
            "game_counter": self.game_count,
            "update_counter": self.update_count,
            "delta_count": self.delta_count,
            "training_time": self.time_elapsed
        }

        torch.save(cp, path)

    def update_target_net(self):
        self.logger("Updating target net")

        self.target_net = copy.deepcopy(self.main_net)
        self.target_net.eval()

    def train(self, batch_size, num_batches, device, shuffle=True):
        for b in range(num_batches):
            batch, idxs, is_weights = self.memory.sample(batch_size, shuffle=shuffle)  # list of tuples ((board, context), action, reward, done)

            self.logger(f"Collecting n-step-return for batch {b}/{num_batches}")

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
                    
            self.logger(f"Calculating TD-errors for batch {b}/{num_batches}")

            discounted_rewards = discounted_rewards.to(device)
            n_step_board = n_step_board.to(device)
            n_step_context = n_step_context.to(device)
            board = board.to(device)
            context = context.to(device)    
            with torch.no_grad():
                state_n_step = (n_step_board, n_step_context)
                action_n_step = self.main_net(state_n_step)
                action_n_step[SingleDeckBot.get_illegal_actions(state_n_step)]
                action_n_step = torch.argmax(action_n_step, dim=1)
                q_n_step = self.target_net(state_n_step)[action_n_step]
                discounted_rewards += (1-dones)*q_n_step*torch.pow(self.discount, actual_n)

            predicted_q = self.main_net((board, context))
            target_q = predicted_q.detach().clone()
            target_q[action] = discounted_rewards

            abs_errors = torch.sum(torch.abs(target_q - predicted_q), dim=1)

            loss = torch.mean((abs_errors**2) * is_weights.to(device))
            self.logger(f"Loss {loss:.4f}, performing backward")

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.memory.update(idxs, abs_errors.detach().cpu())

            self.delta_count += 1
            self.update_count += len(batch)

            if self.delta_count%self.delta == 0:
                self.update_target_net()
    
    def run(self, num_games):
        self.tic = time.time()

        self.main_net = self.main_net.to(self.device)
        self.target_net = self.target_net.to(self.device)

        for i in range(num_games):
            self.logger(f"Starting game {i+1}/{num_games}")

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
            self.logger(f"Finished game {i+1}/{num_games}, total game count {self.game_count}")

            self.memory.add(episodes)
            self.disk_memory.add(episodes)

            self.logger(f"Stored experience, memory-size: {len(self.memory)/self.memory.size}")

            num_batches = len(episodes)//self.batch_size
            if len(episodes)%self.batch_size != 0:
                num_batches += 1

            self.logger(f"Training on new experience for {num_batches} batches")

            self.train(batch_size=self.batch_size, num_batches=num_batches, device=self.device)  # train on new experience
            if self.memory.is_full():
                self.logger("Training on past experience")
                self.train(batch_size=self.batch_size, num_batches=1, device=self.device, shuffle=False)  # train on random experience

            self.logger(f"epsilon decay: {self.eps} -> {self.eps*self.eps_decay}")
            self.eps *= self.eps_decay

            if self.game_count%self.cp_freq == 0:
                self.checkpoint("last.pt")
        
        self.time_elapsed += time.time() - self.tic






