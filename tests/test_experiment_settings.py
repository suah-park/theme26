"""Shared profiles must preserve existing smoke configs and target defaults."""
import importlib.util
import inspect
from pathlib import Path
import tempfile
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('settings_train_entry', ROOT/'scripts/train.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class ExperimentSettingsTests(unittest.TestCase):
    def test_cpu_configs_resolve_shared_profiles(self):
        for path in (ROOT/'configs').glob('e1_*.yaml'):
            with self.subTest(path=path.name):
                config = entry._load_config(path)
                self.assertEqual(config['model']['feature_channels'], 8)
                self.assertEqual(config['model']['num_blocks'], 1)
                self.assertEqual(config['model']['layers_per_block'], 2)
                self.assertEqual(config['model']['growth_rate'], 4)
                self.assertEqual(config['model_reference']['profile'],
                                 config['experiment']['stage'] + '_cpu')

    def test_target_profiles_match_model_constructor_defaults(self):
        from theme26.models.saint import AnisotropicMetaInterpolator, ResidualFusionNetwork
        profiles = yaml.safe_load((ROOT/'configs/model/saint_x4.yaml').read_text())['profiles']
        for stage, model in [('ami', AnisotropicMetaInterpolator), ('rfn', ResidualFusionNetwork)]:
            parameters = inspect.signature(model).parameters
            for key, value in profiles[stage + '_target'].items():
                self.assertEqual(parameters[key].default, value)

    def test_mismatched_stage_and_hidden_model_overrides_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'run.yaml'
            config = {'experiment': {'stage': 'ami'}, 'data': {}, 'training': {},
                      'model': {'settings': 'configs/model/saint_x4.yaml', 'profile': 'rfn_target'}}
            path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, 'match'):
                entry._load_config(path)
            config['model'] = {'settings': 'configs/model/saint_x4.yaml',
                               'profile': 'ami_target', 'feature_channels': 8}
            path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, 'overrides'):
                entry._load_config(path)
