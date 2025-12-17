from __future__ import annotations

import argparse
from pathlib import Path

from mariomind.utils.fs import ensure_project_layout
from mariomind.utils.paths import make_run_id


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="MarioMind",
        description=(
            "CLI central do MarioMind. "
            "Use este arquivo para inicializar pastas, treinar e assistir o agente."
        ),
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Cria a estrutura de pastas (sem gitkeep).")
    p_init.add_argument(
        "--quiet",
        action="store_true",
        help="Não imprime mensagens (útil para scripts).",
    )

    p_train = sub.add_parser("train", help="Treinar um agente.")
    train_sub = p_train.add_subparsers(dest="algo", required=True)

    p_train_ppo = train_sub.add_parser("ppo", help="Treinar PPO (Actor-Critic).")
    p_train_ppo.add_argument("--run-id", type=str, default=None)
    p_train_ppo.add_argument("--num-env", type=int, default=8)

    p_train_dqn = train_sub.add_parser("dqn", help="Treinar Dueling DQN.")
    p_train_dqn.add_argument("--run-id", type=str, default=None)

    p_play = sub.add_parser("play", help="Assistir (render) uma policy/agent.")
    p_play.add_argument("algo", choices=["dqn"], help="Qual agente tocar (por enquanto: dqn).")
    p_play.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Caminho do checkpoint. Ex: runs/<run_id>/models/mario_q_target.pth",
    )
    p_play.add_argument("--sleep", type=float, default=0.001)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # init não deveria depender de gym/torch/etc.
    ensure_project_layout()

    if args.cmd == "init":
        if not args.quiet:
            print("OK: estrutura criada/garantida.")
        return

    if args.cmd == "train":
        run_id = args.run_id or make_run_id(args.algo)

        if args.algo == "ppo":
            from mariomind.algos.ppo import train_ppo  # lazy import

            train_ppo(run_id=run_id, num_env=args.num_env)
            return

        if args.algo == "dqn":
            from mariomind.algos.duel_dqn import train_duel_dqn  # lazy import

            train_duel_dqn(run_id=run_id)
            return

        raise SystemExit("Algoritmo não suportado.")

    if args.cmd == "play":
        if args.algo == "dqn":
            from mariomind.eval.dqn_play import play_dqn  # lazy import

            ckpt = Path(args.ckpt) if args.ckpt else None
            play_dqn(ckpt_path=ckpt, sleep_s=args.sleep)
            return

        raise SystemExit("Modo play não suportado para este algo.")

    raise SystemExit("Comando inválido.")
