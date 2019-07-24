import argparse
from agents import *

FLAGS = None

def main(FLAGS):
    agent = PPO2(FLAGS)

    if FLAGS.train == True:
        print('---- TRAIN ----')
        agent.train(game=FLAGS.game, state=FLAGS.state,num_e=FLAGS.envs)
    if FLAGS.eval == True:
        print('---- EVAL ----')
        agent.evaluate(game=FLAGS.game, state=FLAGS.state)
    if FLAGS.retrain == True:
        print('---- RETRAIN ----')
        agent.retrain(game=FLAGS.game, state=FLAGS.state,num_e=FLAGS.envs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--game',
        type=str,
        default='SonicTheHedgehog-Genesis',
        help='Select Retro environment'
    )

    parser.add_argument(
        '--state',
        nargs='+',
        default='GreenHillZone.Act1',
        help='Select Retro environment Levels'
    )

    parser.add_argument(
        '--eval',
        default=False,
        action='store_true',
        help='Evaluate after training if set'
    )
    parser.add_argument(
        '--train',
        default=False,
        action='store_true',
        help='Train the algorithm if set'
    )
    parser.add_argument(
        '--retrain',
        default=False,
        action='store_true',
        help='Retrain the algorithm if set'
    )
    parser.add_argument(
        '--render',
        default=False,
        action='store_true',
        help='Render the environment if set'
    )
    parser.add_argument(
        '--logdir',
        type=str,
        default='./logs',
        help='Directory to save the tensorboard logfiles'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='./green_hill_1.pkl',
        help='path and name of model file to evaluate'
    )
    parser.add_argument(
        '--envs',
        type=int,
        default=1,
        help='Amount of environments to train simultaneously'
    )
    # execute only if run as the entry point into the program
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)