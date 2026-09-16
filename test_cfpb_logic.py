import pytest
from cfpb_logic import load_artifacts, validate_llm_response, build_input_row

def test_load_artifacts_returns_correct_counts():
  baseline_cols, enriched_cols, llm_categorical_cols, extraction_tool = load_artifacts()
  assert len(baseline_cols) == 8
  assert len(enriched_cols) == 26
  assert len(llm_categorical_cols) == 4
  assert extraction_tool['name'] == 'extract_complaint_features'

def test_valid_repsonse_passes():
  _,_,_, extraction_tool = load_artifacts()
  
  good_response = {
       'harm_type': 'financial_loss',
    'severity': 'high',
    'discrimination_mentioned': False,
    'resolution_requested': 'refund',
    'sentiment_intensity': 'irate'
  }

  assert validate_llm_response(good_response, extraction_tool) is True

def test_invalid_enum_raises_error():
  _,_,_, extraction_tool = load_artifacts()

  bad_response = {
    'harm_type': 'made_up_value',
    'severity': 'high',
    'discrimination_mentioned': False,
    'resolution_requested': 'refund',
    'sentiment_intensity': 'irate'
  }

  with pytest.raises(ValueError):
      validate_llm_response(bad_response, extraction_tool)

def test_missing_field_error():
  _,_,_, extraction_tool = load_artifacts()

  incomplete_response = {'harm_type': 'financial_loss', 'severity': 'high'}

  with pytest.raises(ValueError):
    validate_llm_response(incomplete_response, extraction_tool)

def test_build_input_row_has_correct_key_count():
    llm_features = {
        'harm_type': 'financial_loss',
        'severity': 'high',
        'discrimination_mentioned': False,
        'resolution_requested': 'refund',
        'sentiment_intensity': 'irate'
    }
    row = build_input_row('Credit reporting','Incorrect information',
                          'Information belongs to someone else', 'Experian',
                          'NY', True, True, llm_features)
    assert len(row) == 13

def test_build_input_row_converts_boolean_to_int():
  llm_features = {
       'harm_type': 'financial_loss',
        'severity': 'high',
        'discrimination_mentioned': False,
        'resolution_requested': 'refund',
        'sentiment_intensity': 'irate'
  }
  row = build_input_row('Credit reporting', 'Issue', 'Sub-issue', 'Company',
                        'NY', True, False, llm_features)
  assert row['has_tag'] == 1
  assert row['timely_response'] == 0
