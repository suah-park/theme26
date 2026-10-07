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
    def test_throughput_pilot_preserves_gpu_pilot_settings(self):
        original = yaml.safe_load((ROOT/'configs/experiments/e1_ami_gpu_pilot.yaml').read_text())
        probe = yaml.safe_load((ROOT/'configs/experiments/e1_ami_gpu_throughput_pilot.yaml').read_text())
        self.assertEqual(probe['model'], original['model'])
        self.assertEqual(probe['data'], original['data'])
        for key, value in original['training'].items():
            if key not in ('max_steps_per_epoch', 'max_validation_steps'):
                self.assertEqual(probe['training'][key], value)
        self.assertEqual(probe['training']['max_steps_per_epoch'], 100)
        self.assertEqual(probe['training']['max_validation_steps'], 20)
        self.assertTrue(probe['training']['performance_probe'])
        self.assertNotEqual(probe['output_dir'], original['output_dir'])

    def test_full_epoch1_uses_all_manifest_marginals(self):
        from theme26.data import read_crop_manifest
        target = yaml.safe_load((ROOT/'configs/experiments/e1_ami_target.yaml').read_text())
        full = yaml.safe_load((ROOT/'configs/experiments/e1_ami_full_epoch1.yaml').read_text())
        self.assertEqual(full['model'], target['model'])
        self.assertEqual(full['data'], target['data'])
        self.assertEqual(full['experiment']['seed'], target['experiment']['seed'])
        for key in ('learning_rate', 'mixed_precision', 'loss_region', 'checkpoint_metric'):
            self.assertEqual(full['training'][key], target['training'][key])
        self.assertEqual((full['training']['batch_size'], full['training']['epochs']), (8, 1))
        self.assertIsNone(full['training']['max_steps_per_epoch'])
        self.assertIsNone(full['training']['max_validation_steps'])
        records = read_crop_manifest(ROOT / full['data']['manifest'])
        counts = {split: sum(record.crop.shape_xyz[0] + record.crop.shape_xyz[1]
                             for record in records if record.split == split)
                  for split in ('train', 'validation')}
        self.assertEqual(counts, {'train': 24576, 'validation': 4608})
        self.assertEqual({split: count // 8 for split, count in counts.items()},
                         {'train': 3072, 'validation': 576})

    def test_train_and_inference_share_model_resolution(self):
        import torch
        from theme26.training.workflow import build_stage_model
        for stage in ("ami", "rfn"):
            with self.subTest(stage=stage):
                raw = {"model": {"settings": "configs/model/saint_x4.yaml", "profile": stage + "_cpu"}}
                resolved = entry._load_config(ROOT / f"configs/experiments/smoke/e1_{stage}_synthetic.yaml")
                torch.manual_seed(42)
                train_model = entry._build_model(resolved, stage)
                torch.manual_seed(42)
                inference_model = build_stage_model(raw, stage)
                self.assertEqual(train_model.state_dict().keys(), inference_model.state_dict().keys())
                for key in train_model.state_dict():
                    self.assertTrue(torch.equal(train_model.state_dict()[key], inference_model.state_dict()[key]))

    def test_inference_rejects_hidden_overrides_and_unknown_profiles(self):
        from theme26.training.workflow import build_stage_model
        with self.assertRaisesRegex(ValueError, "overrides"):
            build_stage_model({"model": {"settings": "configs/model/saint_x4.yaml", "profile": "ami_cpu", "feature_channels": 8}}, "ami")
        with self.assertRaisesRegex(ValueError, "unknown model profile"):
            build_stage_model({"model": {"settings": "configs/model/saint_x4.yaml", "profile": "ami_unknown"}}, "ami")

    def test_cpu_configs_resolve_shared_profiles(self):
        for path in (ROOT/'configs/experiments').rglob('e1_*.yaml'):
            settings = yaml.safe_load(path.read_text(encoding='utf-8'))
            if settings['model']['profile'] != settings['experiment']['stage'] + '_cpu':
                continue
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
