"""Field transport checks, without a model or network."""
import copy
import re
import unittest

from rfp_phase4.evidence import ANSWER_SCHEMA
from rfp_phase5.fields import FIELDS, FIELD_PATTERN, FIELD_SCHEMA, parse_field_claim, parse_missing_label


class FieldContractTests(unittest.TestCase):
    def test_eleven_fields_and_source_schema_remain_independent(self):
        self.assertEqual(len(FIELDS), 11)
        self.assertEqual(len({field['id'] for field in FIELDS}), 11)
        self.assertNotIn('pattern', ANSWER_SCHEMA['properties']['claims']['items']['properties']['text'])
        self.assertIn('pattern', FIELD_SCHEMA['properties']['claims']['items']['properties']['text'])

    def test_every_allowed_field_has_exact_category_and_stripped_text(self):
        for field in FIELDS:
            for mode in ['fact', 'notice_reference', 'conflict_candidate']:
                claim = {'category': field['category'], 'text': f'[{field["id"]}|{mode}] 원문에 따른 사실\n조건 포함'}
                original = copy.deepcopy(claim)
                parsed = parse_field_claim(claim)
                self.assertEqual(parsed, {'field_id': field['id'], 'mode': mode, 'text': '원문에 따른 사실\n조건 포함'})
                self.assertTrue(re.fullmatch(FIELD_PATTERN, claim['text']))
                self.assertEqual(claim, original)

    def test_wrong_or_missing_field_metadata_does_not_fall_back_to_keyword_guessing(self):
        for text, category in [
            ('예산은 10원', 'budget'), ('[unknown|fact] 예산', 'budget'),
            ('[budget_vat|fact] 예산', 'overview'), ('[budget_vat|guess] 예산', 'budget'),
            ('[budget_vat|fact] ', 'budget'), ('[budget_vat|fact] [duration|fact] 5개월', 'budget'),
        ]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_field_claim({'category': category, 'text': text})

    def test_missing_details_are_only_bound_to_explicit_known_field(self):
        self.assertEqual(parse_missing_label('[budget_vat] 부가세 포함 여부'),
                         {'field_id': 'budget_vat', 'text': '부가세 포함 여부'})
        self.assertIsNone(parse_missing_label('부가세 포함 여부'))
        self.assertIsNone(parse_missing_label('[unknown] 추정'))
        self.assertIsNone(parse_missing_label('[duration]  '))


if __name__ == '__main__':
    unittest.main()
