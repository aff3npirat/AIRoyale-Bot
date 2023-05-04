# Play a bot vs. bot match.
import os
import subprocess
from multiprocessing import Process, Queue
from argparse import ArgumentParser

import torch

import timing
from device import Controller, ScreenDetector
from utils import seed_all
from bots.single_deck.bot import SingleDeckBot
from constants import TOWER_HP_BOXES, ADB_PATH



def run_bot(
    unit_model,
    side_model,
    number_model,
    deck_names,
    team,
    device,
    queue,
    output,
    eps,
    accept_invite=False,
    network=None,
):
    pid = os.getpid()
    timing.init_logging(os.path.join(output, f"time_{pid}.log"))

    seed_all(pid)

    bot = SingleDeckBot(team, unit_model, number_model, side_model, deck_names, king_levels={"ally": 11, "enemy": 11}, device=device)

    if network is not None:
        bot.init_model(network)

    if accept_invite:
        bot.controller.accept_invite()

    victory = bot.run(eps)

    bot.controller.exit_game()

    experience = bot.with_reward(bot.replay_buffer, victory)
    queue.put(experience)


def play_single_game(output, deck_names, devices, unit_model, side_model, number_model, eps, network):
    TEAMS = ["blue", "red"]

    num_bots = len(devices)

    # send 1v1 invite
    for i in range(0, num_bots, 2):
        Controller(devices[i]).send_clan_1v1()

    # start processes
    out_queue = Queue()
    processes = [Process(target=run_bot, args=(
        unit_model,
        side_model,
        number_model,
        deck_names,
        TEAMS[i],
        devices[i],
        out_queue,
        output,
        eps,
        i%2==1,
        network,
    )) for i in range(num_bots)]

    for p in processes:
        p.start()

    # extract experience from each bot
    experiences = []
    for _ in range(num_bots):
        experiences.extend(out_queue.get())

    for p in processes:
        p.join()
        p.close()

    return experiences


def run(n_games, output, deck_names, devices, unit_model, side_model, number_model, eps, network):
    episodes = []
    for _ in range(n_games):
        episode = play_single_game(
            output=output,
            deck_names=deck_names,
            unit_model=unit_model,
            number_model=number_model,
            side_model=side_model,
            eps=eps,
            devices=devices,
            network=network
        )

        if n_games == 1:
            return episode

        episodes.extend(episode)  # list of tuples (state, action, reward, done)

        for device in devices:
            screen = ScreenDetector(device)
            controller = Controller(device)
            img = controller.take_screenshot()
            while not screen.detect_game_screen(img, "clan"):
                img = controller.take_screenshot()

    return episodes


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("deck", type=str, nargs=8)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--devices", type=int, nargs="+", default=[5555, 5575])
    parser.add_argument("--unit-model", type=str, default="./models/units_singledeck_cpu.onnx")
    parser.add_argument("--num-model", type=str, default="./models/number_cpu.onnx")
    parser.add_argument("--side-model", type=str, default="./models/side_cpu.onnx")
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--net", type=str, default=None)
    parser.add_argument("--n", type=int, default=1, help="number of games to play")

    args = parser.parse_args()

    if len(args.devices)%2 == 1:
        raise ValueError(f"Number of devices must be multiplicative of 2, got {len(args.devices)}")
    
    # initialize adb
    subprocess.run(f"{ADB_PATH} start-server")
    for device in args.devices:
        subprocess.run(f"{ADB_PATH} connect {device}")

    os.makedirs(args.out, exist_ok=True)

    weights = None
    if args.net is not None:
        weights = torch.load(args.net)

    run(
        n_games=args.n,
        output=args.out,
        devices=args.devices,
        deck_names=args.deck,
        unit_model=args.unit_model,
        number_model=args.num_model,
        side_model=args.side_model,
        eps=args.eps,
        network=weights,
    )

    subprocess.run(f"{ADB_PATH} kill-server")
