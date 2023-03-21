# Play a bot vs. bot match.
import os
import logging
import subprocess
from multiprocessing import Process, Queue
from argparse import ArgumentParser

import torch

import timing
import controller
from bots.single_deck.bot import SingleDeckBot



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

    time_logger = logging.getLogger("time")
    time_logger.setLevel(logging.INFO)
    handler_file = logging.FileHandler(os.path.join(output, f"time_{pid}.log"), mode="w+")
    time_logger.addHandler(handler_file)
    time_logger.info("Initialized time logging")

    timing.logger = time_logger

    bot = SingleDeckBot(team, unit_model, number_model, side_model, deck_names, king_levels={"ally": 11, "enemy": 11}, port=port)

    if accept_invite:
        controller.accept_invite(port)

    image = bot.screen.take_screenshot()
    while not bot.in_game(image):
        image = bot.screen.take_screenshot()

    while bot.in_game(image):
        state = bot.get_state(image)
        actions = bot.get_actions(state, eps=eps)
        bot.play_actions(actions)

        bot.store_experience(state, actions)

        image = bot.screen.take_screenshot()

    while not bot.is_game_end(image):
        image = bot.screen.take_screenshot()

    victory = bot.is_victory(image)

    experience = bot.with_reward(bot.replay_buffer, victory)
    queue.put(experience)


def main(output, deck_names, ports, unit_model, side_model, number_model, eps):
    TEAMS = ["blue", "red"]

    num_bots = len(ports)

    subprocess.run([r"..\adb\adb.exe", "start-server"])
    for port in ports:
        subprocess.run([r"..\adb\adb.exe", "connect", f"localhost:{port}"])

    # send 1v1 invite
    for i in range(0, num_bots, 2):
        controller.send_clan_1v1(ports[i])

    # start thread for each bot
    # create bots
    # accept invite
    # start state-action-play loop, while recording experience in replay buffer
    # on loop end: extract rewards from replay buffer
    out_queue = Queue()
    processes = [Process(target=run_bot, args=(unit_model,
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

    # wait for game end
    for p in processes:
        p.join()

    # save on disk
    torch.save(list(out_queue.queue), os.path.join(output, "experience.pt"))


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

    main(
        output=args.out,
        ports=args.ports,
        deck_names=args.deck,
        unit_model=args.unit_model,
        number_model=args.num_model,
        side_model=args.side_model,
        eps=args.eps
    )
