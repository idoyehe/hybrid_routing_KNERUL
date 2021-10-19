import opentuner
from opentuner import ConfigurationManipulator, EnumParameter
from opentuner import MeasurementInterface
from opentuner import Result
from tuning import run_tuning
from common.RL_Envs.rl_env_consts import HyperparamertsStrings
from common.logger import logger
import json


class DdrTuner(MeasurementInterface):
    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """
        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(EnumParameter(HyperparamertsStrings.LEARNING_RATE, [1e-4, 1e-3, 1e-2]))
        manipulator.add_parameter(EnumParameter(HyperparamertsStrings.BATCH_SIZE, [64, 128]))
        manipulator.add_parameter(EnumParameter(HyperparamertsStrings.N_STEPS, [64, 128, 192]))

        manipulator.add_parameter(EnumParameter(HyperparamertsStrings.WEIGHTS_FACTOR, [8, 9, 10, 11, 12]))
        manipulator.add_parameter(EnumParameter(HyperparamertsStrings.WEIGHT_LB, [1e-4, 1e-3, 1e-2]))
        manipulator.add_parameter(EnumParameter(HyperparamertsStrings.WEIGHT_UB, [1, 2, 3]))
        return manipulator

    def run(self, desired_result, input, limit):
        """
        Run training with particular hyperparameters and see how goo the
        performance is
        """
        hyperparamerts = desired_result.configuration.data

        result = run_tuning(hyperparamerts, self.args.config_folder)
        logger.info("Config: {}\nResult:".format(hyperparamerts, result))

        return Result(time=-result)

    def save_final_config(self, configuration):
        """called at the end of tuning"""
        print("Optimal hyperparameters written to hyperparams_final.json:", configuration.data)
        with open('hyperparams_final.json', "w") as outfile:
            json.dump(configuration.data, outfile)


if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    argparser.add_argument('-c', action='store', dest='config_folder', help="Config file to read for the training")
    DdrTuner.main(argparser.parse_args())
