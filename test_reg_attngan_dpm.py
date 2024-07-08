import argparse
from modules.module_reg_attngan_dpm import GAN
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.parsing import AttributeDict
import os

# Configure environment for GPU usage
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class TestOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Adding command-line arguments
        parser.add_argument("--checkpoint_path", default=r"PATH_TO_YOUR_CHECKPOINT")
        parser.add_argument("--test_images_dir", default=r"PATH_TO_YOUR_TEST_IMAGES_DIR")
        parser.add_argument("--output_images_dir", default=r"PATH_TO_YOUR_OUTPUT_IMAGES_DIR")
        parser.add_argument("--batch_size_test", default=1)
        parser.add_argument("--n_workers_test", default=1)
        parser.add_argument("--devices", default="auto")
        parser.add_argument("--accelerator", default="gpu")
        parser.add_argument("--precision_test", default=16)
        parser.add_argument("--is_testing", default=True)

        self.initialized = True
        return parser

    def parse(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt, _ = parser.parse_known_args()
        return opt

if __name__ == "__main__":
    opts = TestOptions()  # Create an instance of TestOptions to parse command-line arguments
    config = AttributeDict(vars(opts.parse()))  # Convert parsed arguments to an AttributeDict for easy access

    # Load the pre-trained GAN model from the specified checkpoint
    gan = GAN.load_from_checkpoint(checkpoint_path=config.checkpoint_path)
    
    # Modify GAN parameters with the configuration options provided
    gan.modify_params(config)
    
    # Create a Trainer instance with the specified configuration
    trainer = Trainer(
        resume_from_checkpoint=config.checkpoint_path,
        accelerator=config.accelerator,
        devices=config.devices,
        precision=config.precision_test,
    )