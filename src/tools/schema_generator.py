import dotenv
dotenv.load_dotenv()
from openai import OpenAI
import json

client = OpenAI()

META_SCHEMA = {
  "name": "metaschema",
  "schema": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "description": "The name of the schema"
      },
      "type": {
        "type": "string",
        "enum": [
          "object",
          "array",
          "string",
          "number",
          "boolean",
          "null"
        ]
      },
      "properties": {
        "type": "object",
        "additionalProperties": {
          "$ref": "#/$defs/schema_definition"
        }
      },
      "items": {
        "anyOf": [
          {
            "$ref": "#/$defs/schema_definition"
          },
          {
            "type": "array",
            "items": {
              "$ref": "#/$defs/schema_definition"
            }
          }
        ]
      },
      "required": {
        "type": "array",
        "items": {
          "type": "string"
        }
      },
      "additionalProperties": {
        "type": "boolean"
      }
    },
    "required": [
      "type"
    ],
    "additionalProperties": False,
    "if": {
      "properties": {
        "type": {
          "const": "object"
        }
      }
    },
    "then": {
      "required": [
        "properties"
      ]
    },
    "$defs": {
      "schema_definition": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "enum": [
              "object",
              "array",
              "string",
              "number",
              "boolean",
              "null"
            ]
          },
          "properties": {
            "type": "object",
            "additionalProperties": {
              "$ref": "#/$defs/schema_definition"
            }
          },
          "items": {
            "anyOf": [
              {
                "$ref": "#/$defs/schema_definition"
              },
              {
                "type": "array",
                "items": {
                  "$ref": "#/$defs/schema_definition"
                }
              }
            ]
          },
          "required": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "additionalProperties": {
            "type": "boolean"
          }
        },
        "required": [
          "type"
        ],
        "additionalProperties": False,
        "if": {
          "properties": {
            "type": {
              "const": "object"
            }
          }
        },
        "then": {
          "required": [
            "properties"
          ]
        }
      }
    }
  }
}

META_PROMPT = """
# Instructions
Return a valid schema for the described JSON.

You must also make sure:
- all fields in an object are set as required
- I REPEAT, ALL FIELDS MUST BE MARKED AS REQUIRED
- all objects must have additionalProperties set to false
    - because of this, some cases like "attributes" or "metadata" properties that would normally allow additional properties should instead have a fixed set of properties
- all objects must have properties defined
- field order matters. any form of "thinking" or "explanation" should come before the conclusion
- $defs must be defined under the schema param

Notable keywords NOT supported include:
- For strings: minLength, maxLength, pattern, format
- For numbers: minimum, maximum, multipleOf
- For objects: patternProperties, unevaluatedProperties, propertyNames, minProperties, maxProperties
- For arrays: unevaluatedItems, contains, minContains, maxContains, minItems, maxItems, uniqueItems

Other notes:
- definitions and recursion are supported
- only if necessary to include references e.g. "$defs", it must be inside the "schema" object

# Examples
Input: Generate a math reasoning schema with steps and a final answer.
Output: {
    "name": "math_reasoning",
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "description": "A sequence of steps involved in solving the math problem.",
            "items": {
                "type": "object",
                "properties": {
                    "explanation": {
                        "type": "string",
                        "description": "Description of the reasoning or method used in this step."
                    },
                    "output": {
                        "type": "string",
                        "description": "Result or outcome of this specific step."
                    }
                },
                "required": [
                    "explanation",
                    "output"
                ],
                "additionalProperties": false
            }
        },
        "final_answer": {
            "type": "string",
            "description": "The final solution or answer to the math problem."
        }
    },
    "required": [
        "steps",
        "final_answer"
    ],
    "additionalProperties": false
}

Input: Give me a linked list
Output: {
    "name": "linked_list",
    "type": "object",
    "properties": {
        "linked_list": {
            "$ref": "#/$defs/linked_list_node",
            "description": "The head node of the linked list."
        }
    },
    "$defs": {
        "linked_list_node": {
            "type": "object",
            "description": "Defines a node in a singly linked list.",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "The value stored in this node."
                },
                "next": {
                    "anyOf": [
                        {
                            "$ref": "#/$defs/linked_list_node"
                        },
                        {
                            "type": "null"
                        }
                    ],
                    "description": "Reference to the next node; null if it is the last node."
                }
            },
            "required": [
                "value",
                "next"
            ],
            "additionalProperties": false
        }
    },
    "required": [
        "linked_list"
    ],
    "additionalProperties": false
}

Input: Dynamically generated UI
Output: {
    "name": "ui",
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "description": "The type of the UI component",
            "enum": [
                "div",
                "button",
                "header",
                "section",
                "field",
                "form"
            ]
        },
        "label": {
            "type": "string",
            "description": "The label of the UI component, used for buttons or form fields"
        },
        "children": {
            "type": "array",
            "description": "Nested UI components",
            "items": {
                "$ref": "#"
            }
        },
        "attributes": {
            "type": "array",
            "description": "Arbitrary attributes for the UI component, suitable for any element",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the attribute, for example onClick or className"
                    },
                    "value": {
                        "type": "string",
                        "description": "The value of the attribute"
                    }
                },
                "required": [
                    "name",
                    "value"
                ],
                "additionalProperties": false
            }
        }
    },
    "required": [
        "type",
        "label",
        "children",
        "attributes"
    ],
    "additionalProperties": false
}
""".strip()

def generate_schema(description: str):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_schema", "json_schema": META_SCHEMA},
        messages=[
            {
                "role": "system",
                "content": META_PROMPT,
            },
            {
                "role": "user",
                "content": "Description:\n" + description,
            },
        ],
    )

    return json.loads(completion.choices[0].message.content)

if __name__ == "__main__":
    print(generate_schema(""" {
        "resume_score_analysis": {
          "type": "object",
          "description": "Detailed scoring and overall job alignment for the resume.",
          "properties": {
            "resume_scoring": {
              "type": "object",
              "description": "Scores and breakdown for the resume scoring.",
              "properties": {
                "scores": {
                  "type": "object",
                  "description": "Individual scores for each section of the resume.",
                  "properties": {
                    "relevance": { "type": "number" },
                    "action_verbs": { "type": "number" },
                    "keywords": { "type": "number" },
                    "formatting": { "type": "number" },
                    "metrics": { "type": "number" },
                    "effectiveness": { "type": "number" },
                    "progression": { "type": "number" },
                    "narrative_consistency": { "type": "number" },
                    "clarity": { "type": "number" },
                    "flow": { "type": "number" },
                    "coherence": { "type": "number" }
                  },
                  "required": [
                    "relevance",
                    "action_verbs",
                    "keywords",
                    "formatting",
                    "metrics",
                    "effectiveness",
                    "progression",
                    "narrative_consistency",
                    "clarity",
                    "flow",
                    "coherence"
                  ],
                  "additionalProperties": false
                },
                "summary_score": {
                "type": "number",
                  "description": "Overall score for the resume."
                }
              },
              "required": ["scores", "summary_score"],
              "additionalProperties": false
            },
            "job_alignment": {
              "type": "object",
              "description": "Overall match score of the job alignment with the resume.",
              "properties": {
                "overall_match_score": { "type": "number" }
              },
              "required": ["overall_match_score"],
              "additionalProperties": false
            }
          },
          "required": ["resume_scoring", "job_alignment"],
          "additionalProperties": false
        }
      }
"""))