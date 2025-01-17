import os
import shutil
import copy
import time
import subprocess
import logging
from argparse import ArgumentParser

import torch

import play
from bots.single_deck.nn import QNet
from bots.single_deck.bot import NEXT_CARD_END, SingleDeckBot
from timing import exec_time
from config import build_options, build_params
from constants import ADB_PATH



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

    def __init__(self, hparams, devices, options, checkpoint=None):
        self.device = options["device"]
        self.cp_freq = options["checkpoint_frequency"]  # number of games
        self.output = options["output"]
        self.disk_memory = options["disk_memory"]

        self.devices = devices
        self.weight_decay = hparams["weight_decay"]
        self.batch_size = hparams["batch_size"]
        self.discount = hparams["discount"]
        self.delta = hparams["delta"]
        self.eps_decay = hparams["eps_decay"]
        self.n = hparams["n"]
        self.deck_names = hparams["deck_names"]
        self.unit_model = hparams["unit_model"]
        self.side_model = hparams["side_model"]
        self.number_model = hparams["number_model"]

        if checkpoint is None:
            self.eps = hparams["eps0"]
            lr = hparams["lr0"]
            self.game_count = 0
            self.sample_count = 0
            self.update_count = 0
            self.memory = hparams["memory"]
            self.time_elapsed = 0
        else:
            cp = torch.load(checkpoint, map_location="cpu")

            self.memory = cp["memory"]

            self.eps = cp["eps"]
            lr = cp["lr"]

            main_net_state_dict = cp["main_net"]
            target_net_state_dict = cp["target_net"]

            self.game_count = cp["game_counter"]
            self.sample_count = cp["update_counter"]
            self.update_count = cp["delta_count"]
            self.time_elapsed = cp["training_time"]

            logging.info(f"Loaded checkpoint '{checkpoint}'")

        self.main_net = QNet([512+NEXT_CARD_END, 128, 64, 5], activation=params["activation"], bias=True, feature_extractor=False)
        if checkpoint is not None:
            self.main_net.load_state_dict(main_net_state_dict)
        self.main_net.train()

        self.target_net = copy.deepcopy(self.main_net)
        if checkpoint is not None:
            self.target_net.load_state_dict(target_net_state_dict)
        self.target_net.eval()

        self.optim = torch.optim.SGD(self.main_net.parameters(), lr=lr, weight_decay=self.weight_decay)
        self.lr_decay = hparams["lr_decay"](self.optim)
        self.lr_decay.step(self.game_count)

    def checkpoint(self, name):
        path = os.path.join(self.output, "checkpoints")
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=False)

        path = os.path.join(path, name)

        self.time_elapsed += time.time() - self.tic
        self.tic = time.time()

        cp = {
            "main_net": self.main_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "memory": self.memory,
            "eps": self.eps,
            "lr": self.optim.param_groups[0]["lr"],
            "game_counter": self.game_count,
            "update_counter": self.sample_count,
            "delta_count": self.update_count,
            "training_time": self.time_elapsed
        }

        torch.save(cp, path)

    def update_target_net(self):
        logging.info("Updating target net")

        self.target_net = copy.deepcopy(self.main_net)
        self.target_net.eval()

    @exec_time
    def train(self, batch_size, num_batches, device, shuffle=True):
        self.main_net = self.main_net.to(self.device)
        self.target_net = self.target_net.to(self.device)

        for b in range(num_batches):
            batch, idxs, is_weights = self.memory.sample(batch_size, shuffle=shuffle)  # list of tuples ((board, context), action, reward, done)

            board = torch.stack([batch[i][0][0] for i in range(batch_size)], dim=0)
            context = torch.stack([batch[i][0][1] for i in range(batch_size)], dim=0)
            action = torch.tensor([batch[i][1] for i in range(batch_size)], dtype=torch.int64)

            logging.info(f"Collecting n-step-return for batch {b}/{num_batches}")

            n_step_board = torch.zeros((batch_size, *batch[0][0][0].shape))
            n_step_context = torch.zeros((batch_size, *batch[0][0][1].shape))
            dones = torch.empty(batch_size)
            discounted_rewards = torch.empty(batch_size)
            actual_n = torch.zeros(batch_size)
            for i in range(batch_size):
                discounted_reward, last_idx = n_step_return(self.n, self.memory, idxs[i], self.discount)
                discounted_rewards[i] = discounted_reward
                
                if self.memory[last_idx][3]:  # done
                    dones[i] = 1.0
                else:
                    state_n_step = self.memory[(last_idx+1)%len(self.memory)][0]
                    
                    n_step_board[i] = state_n_step[0]
                    n_step_context[i] = state_n_step[1]
                    dones[i] = 0.0
                    actual_n[i] = self.n
                    
            logging.info(f"Calculating TD-errors for batch {b}/{num_batches}")

            discounted_rewards = discounted_rewards.to(device)
            n_step_board = n_step_board.to(device)
            n_step_context = n_step_context.to(device)
            board = board.to(device)
            context = context.to(device)
            with torch.no_grad():
                action_n_step = self.main_net((n_step_board, n_step_context))
                #  mask out all illegal actions
                for i in range(batch_size):
                    action_n_step[i][SingleDeckBot.get_illegal_actions((n_step_board[i], n_step_context[i]))] = -torch.inf

                action_n_step = torch.argmax(action_n_step, dim=1)
                q_n_step = self.target_net((n_step_board, n_step_context))[torch.arange(batch_size), action_n_step]
                discounted_rewards += (1-dones.to(self.device))*q_n_step*torch.pow(self.discount, actual_n).to(self.device)

            predicted_q = self.main_net((board, context))
            target_q = predicted_q.detach().clone()
            target_q[list(range(batch_size)), action] = discounted_rewards

            abs_errors = torch.sum(torch.abs(target_q - predicted_q), dim=1)
            logging.info(f"TD errors: {abs_errors.mean().item():.4f}+-{abs_errors.std().item():.4f} | min={abs_errors.min().item():.2g}, max={abs_errors.max():.2g}")

            loss = torch.mean((abs_errors**2) * is_weights.to(device))
            logging.info(f"Loss {loss:.4f}, performing backward")

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.memory.update(idxs, abs_errors.detach().cpu())

            self.update_count += 1
            self.sample_count += len(batch)

            # log mean difference between main and target net, should always be greater 0 otherwise no learning is done
            with torch.no_grad():
                param_main = dict(self.main_net.named_parameters())
                param_target = dict(self.target_net.named_parameters())
                for n in param_main:
                    mean_diff = torch.mean(torch.abs(param_main[n] - param_target[n]))
                    mean_grad = torch.mean(param_main[n].grad)
                    std_grad = torch.std(param_main[n].grad)
                    logging.info(f"[{n}] mean-diff: {mean_diff.item():.3g} | grad: {mean_grad.item():.3g}+-{std_grad.item():.3g}")
        
            if self.update_count%self.delta == 0:
                self.update_target_net()
    
    def run(self, num_games, random_bot=False):
        self.tic = time.time()

        for i in range(num_games):
            logging.info(f"Starting game {i+1}/{num_games}")

            episodes = play.run(
                n_games=1,
                deck_names=self.deck_names,
                devices=self.devices,
                unit_model=self.unit_model,
                side_model=self.side_model,
                number_model=self.number_model,
                eps=self.eps,
                network=self.main_net.cpu().state_dict(),
                random_bot=random_bot,
            )

            self.game_count += 1
            logging.info(f"Finished game {i+1}/{num_games}, total game count {self.game_count}")

            logging.info("Storing in replay memory...")
            self.memory.add(episodes)
            logging.info("Storing on disk...")
            self.disk_memory.add(episodes)

            logging.info(f"Stored {len(episodes)} experiences, memory-size: {len(self.memory)}/{self.memory.size}")

            N = min(len(episodes), self.memory.size)
            logging.info(f"Training on {N} new experiences")

            num_batches = N//self.batch_size
            logging.info(f"Training on {num_batches*self.batch_size} samples...")
            self.train(batch_size=self.batch_size, num_batches=num_batches, device=self.device)  # train on new experience

            partial_batch = N%self.batch_size
            if partial_batch > 0:
                logging.info(f"Training on {partial_batch} samples...")
                self.train(batch_size=partial_batch, num_batches=1, device=self.device)  # train on new experience

            if self.memory.is_full():
                logging.info("Training on past experience")
                self.train(batch_size=self.batch_size, num_batches=1, device=self.device, shuffle=False)  # train on random experience

                self.eps *= self.eps_decay
                if self.eps < 0.001:
                    self.eps = 0
                logging.info(f"New epsilon: {self.eps}")

                self.lr_decay.step(self.game_count)
                logging.info(f"New learningrate: {self.optim.param_groups[0]['lr']}")

            if self.game_count%self.cp_freq == 0:
                self.checkpoint(f"game_{self.game_count}.pt")
            self.checkpoint("last.pt")
        
        self.time_elapsed += time.time() - self.tic


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--opt", type=str)
    parser.add_argument("--params", type=str)
    parser.add_argument("--n", type=int)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--devices", nargs=2, type=str)
    parser.add_argument("--random", action="store_true")

    args = parser.parse_args()

    options = build_options(opts_file=args.opt)
    params = build_params(params_file=args.params)

    out_dir = options["output"]
    os.makedirs(out_dir, exist_ok=True)

    if os.path.abspath(os.path.join(args.opt, "..")) != os.path.abspath(out_dir):
        shutil.copy(args.opt, os.path.join(out_dir, "options.yaml"))
    if os.path.abspath(os.path.join(args.params, "..")) != os.path.abspath(out_dir):
        shutil.copy(args.params, os.path.join(out_dir, "hparams.yaml"))

    subprocess.run(f"{ADB_PATH} start-server")
    for device in args.devices:
        subprocess.run(f"{ADB_PATH} connect {device}")

    trainer = Trainer(
        hparams=params,
        options=options,
        devices=args.devices,
        checkpoint=args.resume,
    )

    trainer.run(args.n, random_bot=args.random)

    subprocess.run(f"{ADB_PATH} kill-server")
