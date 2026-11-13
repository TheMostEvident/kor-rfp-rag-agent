"""Notebook diagnostics must fail before science imports or API activity."""
from importlib.metadata import PackageNotFoundError
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rfp_phase4.runtime import EnvironmentMismatch, bootstrap, environment_report, source_allowlist


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'index').mkdir()
        self.expected = {'numpy':'2.3.5','scipy':'1.17.0','scikit-learn':'1.8.0'}
        (self.root/'index/index_meta.json').write_text(json.dumps({'dependencies':self.expected}))

    def test_matching_environment_is_compatible_without_path_assumptions(self):
        with patch('rfp_phase4.runtime.version', side_effect=self.expected.__getitem__):
            result = environment_report(self.root)
        self.assertTrue(result['compatible'])
        self.assertFalse(result['automatic_install_or_rebuild'])

    def test_wrong_environment_has_specific_versions_and_actionable_kernel_hint(self):
        with patch('rfp_phase4.runtime.version', return_value='different'):
            result = environment_report(self.root)
        self.assertEqual(set(result['mismatches']),set(self.expected))
        self.assertIn('.venv/bin/python', result['kernel_hint'])
        self.assertIn('requirements-phase4.txt', result['install_hint'])

    def test_missing_dependency_is_reported_without_import(self):
        def get(name):
            if name=='scipy': raise PackageNotFoundError(name)
            return self.expected[name]
        with patch('rfp_phase4.runtime.version', side_effect=get):
            result = environment_report(self.root)
        self.assertIsNone(result['current_dependencies']['scipy'])
        self.assertIn('scipy', result['mismatches'])

    def test_bootstrap_mismatch_precedes_index_source_or_api_loading(self):
        with patch('rfp_phase4.runtime.version', return_value='different'):
            with self.assertRaises(EnvironmentMismatch) as raised:
                bootstrap(self.root)
        self.assertIn('노트북 Python:',str(raised.exception))
        self.assertIn('OpenAI', str(raised.exception))

    def test_missing_package_has_clear_error(self):
        (self.root/'index/index_meta.json').unlink()
        with self.assertRaisesRegex(FileNotFoundError,'전체 Phase 4 ZIP'):
            environment_report(self.root)

    def test_allowlist_is_user_selected_and_canonical(self):
        class Index:
            def resolve_doc_ids(self, ids):
                if ids!=['alias']: raise ValueError('unknown')
                return ['canonical']
            def get_catalog(self): return [{'doc_id':'canonical','sha256':'a'*64}]
        self.assertEqual(source_allowlist(Index(),['alias']),{'canonical':'a'*64})
        with self.assertRaises(ValueError): source_allowlist(Index(),[])
        with self.assertRaises(ValueError): source_allowlist(Index(),['unknown'])


if __name__ == '__main__':
    unittest.main()
