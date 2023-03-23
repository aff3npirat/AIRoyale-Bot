# Play a bot vs. bot match.
import os
import time
import subprocess
from multiprocessing import Process, Queue
from argparse import ArgumentParser

import torch

import timing
import controller
from utils import seed_all
from bots.single_deck.bot import SingleDeckBot
from constants import TOWER_HP_BOXES, PRINCESS_Y_OFFSET, ADB_PATH



def run_bot(
    unit_model,
    side_model,
    number_model,
    deck_names,
    team,
    port,
    queue,
    output,
    eps,
    accept_invite=False
):
    os.makedirs(output, exist_ok=True)

    pid = os.getpid()
    timing.init_logging(os.path.join(output, f"time_{pid}.log"))

    seed_all(pid)

    for i, (name, (x1, y1, x2, y2)) in enumerate(TOWER_HP_BOXES):
        if "princess" in name:
            TOWER_HP_BOXES[i][1] = (x1, y1-PRINCESS_Y_OFFSET, x2, y2-PRINCESS_Y_OFFSET)

    bot = SingleDeckBot(team, unit_model, number_model, side_model, deck_names, king_levels={"ally": 11, "enemy": 11}, port=port)

    if accept_invite:
        controller.accept_invite(bot.screen)

    victory = bot.run(eps)

    controller.exit_game(bot.screen)

    experience = bot.with_reward(bot.replay_buffer, victory)
    queue.put(experience)


def main(output, deck_names, ports, unit_model, side_model, number_model, eps):
    TEAMS = ["blue", "red"]

    num_bots = len(ports)

    # send 1v1 invite
    for i in range(0, num_bots, 2):
        controller.send_clan_1v1(ports[i])

    # start processes
    out_queue = Queue()
    processes = [Process(target=run_bot, args=(
        unit_model,
        side_model,
        number_model,
        deck_names,
        TEAMS[i],
        ports[i],
        out_queue,
        output,
        eps,
        i%2==1
    )) for i in range(num_bots)]

    for p in processes:
        p.start()

    # extract experience from each bot
    experiences = []
    for _ in range(num_bots):
        experiences.extend(out_queue.get())
    torch.save(experiences, os.path.join(output, "experience.pt"))

    for p in processes:
        p.join()
        p.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("deck", type=str, nargs=8)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--ports", type=int, nargs="+", default=[5555, 5575])
    parser.add_argument("--unit-model", type=str, default="./models/units_singledeck_cpu.onnx")
    parser.add_argument("--num-model", type=str, default="./models/number_cpu.onnx")
    parser.add_argument("--side-model", type=str, default="./models/side_cpu.onnx")
    parser.add_argument("--eps", type=float, default=0.0)

    args = parser.parse_args()

    if len(args.ports)%2 == 1:
        raise ValueError(f"Number of ports must be multiplicative of 2, got {len(args.ports)}")
    
    # initialize adb
    subprocess.run(f"{ADB_PATH} start-server")
    for port in args.ports:
        subprocess.run(f"{ADB_PATH} connect localhost:{port}")

    main(
        output=args.out,
        ports=args.ports,
        deck_names=args.deck,
        unit_model=args.unit_model,
        number_model=args.num_model,
        side_model=args.side_model,
        eps=args.eps
    )
