import json
import pandas as pd

def load_artifacts():
  '''Load the core JSON artifacts used by the cfpb prediction pipeline'''
  with open('baseline_cols.json') as f:
    baseline_cols = json.load(f)
  with open('enriched_cols.json') as f:
    enriched_cols = json.load(f)
  with open('llm_categorical_cols.json') as f:
    llm_categorical_cols = json.load(f)
  with open('extraction_tool.json') as f:
    extraction_tool = json.load(f)
  return baseline_cols, enriched_cols, llm_categorical_cols, extraction_tool

def validate_llm_response(response:dict, extraction_tool:dict)-> bool:
  '''Check that an LLM response has a;; the required fields with allowed enum vlaues'''
  schema = extraction_tool['input_schema']['properties']
  required = extraction_tool['input_schema']['required']

  for field in required:
    if field not in response:
      raise ValueError(f"Missing rquired field {field}")
    
    if 'enum' in schema[field] and response[field] not in schema[field]['enum']:
      raise ValueError(f"Invalid value for {field}: {response[field]}")
  return True

def build_input_row(product: str, issue: str, sub_issue: str, company: str,
                    state: str, has_tag: bool, timely_response: bool,
                    llm_features: dict)-> dict:
    '''Combine structured from inputs and LLM features into one row dict.'''
    row = {
        'product_clean': product,
        'Issue': issue,
        'Sub-issue' : sub_issue,
        'Company': company,
        'State' : state,
        'has_tag': int(has_tag),
        'has_narrative': 1,
        'timely_response': int(timely_response)
    }
    row.update(llm_features)
    return row
